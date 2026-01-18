from ultralytics import YOLO

# โหลดโมเดล YOLO11n ที่คุณเทรนมา
model = YOLO('best.pt') 

# สั่งตรวจจับภาพ (เปลี่ยน path ภาพตามต้องการ)
results = model.predict(source='your_fruit_image.jpg', save=True)
