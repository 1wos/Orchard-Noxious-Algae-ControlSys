"""
Raspberry Pi 엣지 추론 성능 비교 스크립트
- CPU only (YOLOv8)
- Intel Movidius NCS (OpenVINO)
- Google Coral USB Accelerator (EdgeTPU)

Usage:
    python rasb_performance_comparison.py \
        --video ./video/input.mp4 \
        --model-dir ./models \
        --output ./output \
        --mode cpu          # cpu | movidius | coral
"""

import argparse
import os
import time
import psutil
import csv
import cv2
import matplotlib.pyplot as plt
from gpiozero import CPUTemperature
from ultralytics import YOLO

# Coral USB Accelerator
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import detect


def parse_args():
    parser = argparse.ArgumentParser(description="RPi 엣지 추론 성능 비교")
    parser.add_argument("--video", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--model-dir", type=str, required=True, help="모델 디렉토리 경로")
    parser.add_argument("--output", type=str, default="./output", help="결과 저장 디렉토리")
    parser.add_argument("--mode", type=str, default="cpu", choices=["cpu", "movidius", "coral"],
                        help="추론 모드: cpu | movidius | coral")
    parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    # 모드별 출력 디렉토리
    mode_dir_map = {
        "cpu": "Raspberry_Pi_only",
        "movidius": "Raspberry_Pi_Movidius_TPU",
        "coral": "Raspberry_Pi_Coral_USB_Accelerator",
    }
    output_base_dir = os.path.join(args.output, mode_dir_map[args.mode])
    os.makedirs(output_base_dir, exist_ok=True)

    # 모델 파일 검색
    ext = (".pt", ".tflite") if args.mode != "movidius" else (".xml",)
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith(ext)]

    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        output_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        output_video_path = os.path.join(output_dir, "result_video.mp4")
        results_csv_path = os.path.join(output_dir, "results.csv")
        plots_separate_path = os.path.join(output_dir, "plots_separate.png")
        plots_combined_path = os.path.join(output_dir, "plots_combined.png")

        # 모드별 모델 로드
        if args.mode == "cpu":
            model = YOLO(model_path)
        elif args.mode == "movidius":
            from openvino.inference_engine import IECore
            ie = IECore()
            net = ie.read_network(model=model_path)
            exec_net = ie.load_network(network=net, device_name="MYRIAD")
        elif args.mode == "coral":
            interpreter = make_interpreter(model_path)
            interpreter.allocate_tensors()

        memory_usage_start = psutil.Process().memory_info().rss / 1024 / 1024

        cap = cv2.VideoCapture(args.video)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

        results_data = []
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cpu_usage_start = psutil.cpu_percent(interval=None)

            # --- 추론 ---
            if args.mode == "cpu":
                inference_start = time.time()
                results = model(frame)[0]
                inference_time = time.time() - inference_start

            elif args.mode == "movidius":
                input_blob = next(iter(net.input_info))
                out_blob = next(iter(net.outputs))
                n, c, h, w = net.input_info[input_blob].input_data.shape
                image = cv2.resize(frame, (w, h)).transpose((2, 0, 1)).reshape((n, c, h, w))
                inference_start = time.time()
                res = exec_net.infer(inputs={input_blob: image})
                inference_time = time.time() - inference_start
                results = res[out_blob]

            elif args.mode == "coral":
                inference_start = time.time()
                _, scale = common.set_resized_input(
                    interpreter, (frame.shape[1], frame.shape[0]),
                    lambda size: cv2.resize(frame, size))
                interpreter.invoke()
                results = detect.get_objects(interpreter, args.threshold, scale)
                inference_time = time.time() - inference_start

            # --- 메트릭 수집 ---
            cpu_usage = (cpu_usage_start + psutil.cpu_percent(interval=None)) / 2
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - memory_usage_start
            temperature = CPUTemperature().temperature

            # --- bbox 그리기 ---
            if args.mode in ("cpu", "movidius"):
                for result in results:
                    x1, y1, x2, y2, score, class_id = result
                    if score > args.threshold:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, results.names[int(class_id)].upper(),
                                    (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            elif args.mode == "coral":
                for result in results:
                    bbox = result.bbox
                    if result.score > args.threshold:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"Class: {result.id}, Score: {result.score:.2f}",
                                    (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            out.write(frame)
            results_data.append([int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                 inference_time, memory_usage, temperature, cpu_usage])

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        print(f"[{model_name}] Total: {total_time:.2f}s, Frames: {len(results_data)}")

        # CSV 저장
        with open(results_csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Inference Time (s)", "Memory Usage (MB)",
                             "CPU Temperature (°C)", "CPU Usage (%)"])
            writer.writerows(results_data)

        # 개별 플롯
        times = [x[1] for x in results_data]
        memory_usages = [x[2] for x in results_data]
        cpu_temperatures = [x[3] for x in results_data]
        cpu_usages = [x[4] for x in results_data]

        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        for ax, data, title, color, ylabel in zip(
            axes,
            [times, memory_usages, cpu_temperatures, cpu_usages],
            ["Inference Time", "Memory Usage", "CPU Temperature", "CPU Usage"],
            ["r", "g", "b", "m"],
            ["Time (s)", "MB", "°C", "%"],
        ):
            ax.plot(data, f"{color}-")
            ax.set_title(f"{title} per Frame")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plots_separate_path)
        plt.close()

        # 통합 플롯
        plt.figure(figsize=(10, 6))
        plt.plot(times, "r-", label="Inference Time (s)")
        plt.plot(memory_usages, "g-", label="Memory Usage (MB)")
        plt.plot(cpu_temperatures, "b-", label="CPU Temperature (°C)")
        plt.plot(cpu_usages, "m-", label="CPU Usage (%)")
        plt.title("Performance Metrics per Frame")
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_combined_path)
        plt.close()


if __name__ == "__main__":
    main()
