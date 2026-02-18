from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
model = YOLO("H:/mobile_used_project/runs/detect/train2/weights/best.pt")
#"H:/mobile_used_project/test/lib2.mp4"
# "https://192.168.1.109:8080/video"
vidu = model(source="https://192.168.128.87:8080/video",conf=0.6, show=True, save=True,device=0)
# predict_img
# im = Image.open("H:/mobile_used_project/mobile_used_dataset/val/images/mp_53.png")
# y_prob = model.predict(source=im, save=True,device=0)

#run in gpu
# print(rs_test[0].names)
# print(rs_test[0].probs)

# for result in rs_test:
#     probs = result.probs # Probs object for classification outputs
# print(probs)

# for result in rs_test:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     confs = boxes.data[:, 4:6]  # Confidence and class ID of the detected objects
# print(confs)

# print(torch._C._cuda_getDeviceCount() > 0)

# print(torch.version.cuda)




# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())
# print("Number of GPUs:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("Device name:", torch.cuda.get_device_name(0))
