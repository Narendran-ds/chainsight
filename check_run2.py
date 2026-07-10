
from ultralytics import YOLO

def main():
    model = YOLO(r"E:\Nari\Naren\chainsight\models\finetuned\run2_exit_marker\weights\best.pt")
    metrics = model.val(data=r"E:\Nari\Naren\chainsight\data\processed\chainsight_dataset\data.yaml")

    print("\n--- Per-class AP50 ---")
    for class_id, name in model.names.items():
        print(f"{name:<25} AP50: {metrics.box.ap50[class_id]:.4f}")

    print(f"\nOverall mAP50: {metrics.box.map50:.4f}")
    print(f"Overall mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()