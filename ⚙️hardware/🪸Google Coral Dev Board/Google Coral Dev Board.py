"""
Google Coral Dev Board EdgeTPU 추론 스크립트
- PyCoral API로 TFLite 모델 추론
- CPU 온도, 메모리 사용량 모니터링

Usage:
    python "Google Coral Dev Board.py" \
        --video ./video/input.mp4 \
        --model-dir ./models \
        --output ./output
"""

import argparse
import os
import time
import psutil
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter


def get_cpu_temperature():
    try:
        temp = os.popen("cat /sys/class/thermal/thermal_zone0/temp").readline()
        return float(temp) / 1000
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Google Coral Dev Board EdgeTPU 추론")
    parser.add_argument("--video", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--model-dir", type=str, required=True, help="모델 디렉토리 (.tflite)")
    parser.add_argument("--output", type=str, default="./output", help="결과 저장 디렉토리")
    parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model_files = [f for f in os.listdir(args.model_dir) if f.endswith(".tflite")]

    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        output_dir = os.path.join(args.output, model_name)
        os.makedirs(output_dir, exist_ok=True)

        output_video_path = os.path.join(output_dir, "result_video.mp4")
        results_csv_path = os.path.join(output_dir, "results.csv")
        plots_separate_path = os.path.join(output_dir, "plots_separate.png")
        plots_combined_path = os.path.join(output_dir, "plots_combined.png")

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

            # 추론
            inference_start = time.time()
            _, scale = common.set_resized_input(
                interpreter, (frame.shape[1], frame.shape[0]),
                lambda size: cv2.resize(frame, size))
            interpreter.invoke()
            results = detect.get_objects(interpreter, args.threshold, scale)
            inference_time = time.time() - inference_start

            # 메트릭 수집
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - memory_usage_start
            cpu_temperature = get_cpu_temperature()
            gpu_temperature = None

            # bbox 그리기
            for result in results:
                bbox = result.bbox
                if result.score > args.threshold:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"Class: {result.id}, Score: {result.score:.2f}",
                                (int(x1), int(y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            out.write(frame)
            results_data.append([
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                inference_time, memory_usage,
                cpu_temperature, gpu_temperature,
            ])

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        print(f"[{model_name}] Total: {total_time:.2f}s, Frames: {len(results_data)}")

        # CSV 저장
        with open(results_csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Inference Time (s)", "Memory Usage (MB)",
                             "CPU Temperature (°C)", "GPU Temperature (°C)"])
            writer.writerows(results_data)

        # 개별 플롯
        times = [x[1] for x in results_data]
        memory_usages = [x[2] for x in results_data]
        cpu_temperatures = [x[3] for x in results_data]
        gpu_temperatures = [x[4] for x in results_data] if results_data[0][4] is not None else None

        num_plots = 4 if gpu_temperatures else 3
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 10))
        for ax, data, title, color, ylabel in zip(
            axes,
            [times, memory_usages, cpu_temperatures] + ([gpu_temperatures] if gpu_temperatures else []),
            ["Inference Time", "Memory Usage", "CPU Temperature"] + (["GPU Temperature"] if gpu_temperatures else []),
            ["r", "g", "b"] + (["c"] if gpu_temperatures else []),
            ["Time (s)", "MB", "°C"] + (["°C"] if gpu_temperatures else []),
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
        if gpu_temperatures:
            plt.plot(gpu_temperatures, "c-", label="GPU Temperature (°C)")
        plt.title("Performance Metrics per Frame")
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_combined_path)
        plt.close()


if __name__ == "__main__":
    main()
