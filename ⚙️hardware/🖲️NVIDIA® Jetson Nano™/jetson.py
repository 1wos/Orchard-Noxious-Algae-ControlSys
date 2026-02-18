"""
Jetson Nano 엣지 추론 + GPU 모니터링 스크립트
- jtop으로 CPU/GPU 온도, GPU 사용률 추적
- YOLOv8 모델 비디오 추론

Usage:
    python jetson.py \
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
import matplotlib.pyplot as plt
from ultralytics import YOLO
from jtop import jtop


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano 엣지 추론 벤치마크")
    parser.add_argument("--video", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--model-dir", type=str, required=True, help="모델 디렉토리 (.pt)")
    parser.add_argument("--output", type=str, default="./output", help="결과 저장 디렉토리")
    parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model_files = [f for f in os.listdir(args.model_dir) if f.endswith(".pt")]

    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        output_dir = os.path.join(args.output, model_name)
        os.makedirs(output_dir, exist_ok=True)

        output_video_path = os.path.join(output_dir, "result_video.mp4")
        results_csv_path = os.path.join(output_dir, "results.csv")
        plots_separate_path = os.path.join(output_dir, "plots_separate.png")
        plots_combined_path = os.path.join(output_dir, "plots_combined.png")

        model = YOLO(model_path)
        memory_usage_start = psutil.Process().memory_info().rss / 1024 / 1024

        cap = cv2.VideoCapture(args.video)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

        results_data = []
        start_time = time.time()

        with jtop() as jetson:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cpu_usage_start = psutil.cpu_percent(interval=None)

                # 추론
                inference_start = time.time()
                results = model(frame)[0]
                inference_time = time.time() - inference_start

                # 메트릭 수집
                cpu_usage = (cpu_usage_start + psutil.cpu_percent(interval=None)) / 2
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - memory_usage_start
                cpu_temperature = jetson.temperature["CPU"]
                gpu_temperature = jetson.temperature["GPU"]
                gpu_usage = jetson.gpu["val"]

                # bbox 그리기
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    if score > args.threshold:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, results.names[int(class_id)].upper(),
                                    (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                out.write(frame)
                results_data.append([
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    inference_time, memory_usage,
                    cpu_temperature, gpu_temperature,
                    cpu_usage, gpu_usage,
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
                             "CPU Temperature (°C)", "GPU Temperature (°C)",
                             "CPU Usage (%)", "GPU Usage (%)"])
            writer.writerows(results_data)

        # 개별 플롯
        metrics = {
            "Inference Time": ([x[1] for x in results_data], "r", "Time (s)"),
            "Memory Usage": ([x[2] for x in results_data], "g", "MB"),
            "CPU Temperature": ([x[3] for x in results_data], "b", "°C"),
            "GPU Temperature": ([x[4] for x in results_data], "c", "°C"),
            "CPU Usage": ([x[5] for x in results_data], "m", "%"),
            "GPU Usage": ([x[6] for x in results_data], "y", "%"),
        }

        fig, axes = plt.subplots(6, 1, figsize=(10, 12))
        for ax, (title, (data, color, ylabel)) in zip(axes, metrics.items()):
            ax.plot(data, f"{color}-")
            ax.set_title(f"{title} per Frame")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(plots_separate_path)
        plt.close()

        # 통합 플롯
        plt.figure(figsize=(10, 8))
        for title, (data, color, _) in metrics.items():
            plt.plot(data, f"{color}-", label=title)
        plt.title("Performance Metrics per Frame")
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_combined_path)
        plt.close()


if __name__ == "__main__":
    main()
