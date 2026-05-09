"""
train.py
Day5: YOLOv8 目标检测训练
"""
from ultralytics import YOLO


def main():
    # 加载预训练的 YOLOv8n（nano，最轻量，适合笔记本 GPU）
    # 其他可选: yolov8s.pt (small), yolov8m.pt (medium)
    print("[模型] 加载 YOLOv8n 预训练权重...")
    model = YOLO('yolov8n.pt')

    # 训练
    # imgsz=640: 输入尺寸 640x640
    # epochs=20: 训练 20 轮
    # batch=16: 每批 16 张（4070 8G 显存可以承受）
    # device=0: 使用 GPU
    print("[训练] 开始训练...")
    results = model.train(
        data='data.yaml',
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project='./runs',
        name='yolo_v8n_detection',
        exist_ok=True,
        verbose=True
    )

    print("\n✅ 训练完成！")
    print(f"最佳 mAP@0.5: {results.results_dict.get('metrics/mAP50', 'N/A')}")
    print(f"最佳 mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95', 'N/A')}")
    print("查看结果: runs/yolo_v8n_detection/")


if __name__ == "__main__":
    main()