import onnxruntime as ort
import numpy as np
import cv2
import os

class HumanSeg:
    def __init__(self, use_fp16=True, cache_dir=None):
        """
        Khởi tạo thư viện JetSeg.
        :param use_fp16: Bật chế độ FP16 (Nhanh gấp đôi trên Jetson).
        :param cache_dir: Tùy chỉnh nơi lưu cache. Nếu None, sẽ dùng ~/.cache/jetseg
        """
        # 1. Tự động định vị file model bên trong thư viện
        current_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(current_dir, "human_seg.onnx")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ CRITICAL: Không tìm thấy model tại {self.model_path}")

        self.input_size = (224, 224)
        
        # 2. CẤU HÌNH CACHE TẬP TRUNG (Fixed)
        if cache_dir is None:
            # Lấy đường dẫn Home của User (ví dụ: /home/orin/)
            home_dir = os.path.expanduser("~")
            # Tạo đường dẫn chuẩn: /home/orin/.cache/jetseg
            cache_dir = os.path.join(home_dir, ".cache", "jetseg")
        
        # Tạo thư mục nếu chưa có
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                print(f"📂 [JetSeg] Đã tạo thư mục cache mới: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Không thể tạo cache tại {cache_dir}. Dùng thư mục tạm.")
                cache_dir = "/tmp/jetseg_cache"
                os.makedirs(cache_dir, exist_ok=True)
        else:
             print(f"📂 [JetSeg] Sử dụng cache tại: {cache_dir}")

        # 3. Cấu hình TensorRT Provider
        trt_options = {
            'trt_fp16_enable': use_fp16,
            'trt_int8_enable': False,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir, # Trỏ về cache tập trung
            'trt_max_workspace_size': 2147483648, # 2GB RAM build engine
        }

        # Cấu hình Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 4. Tìm Provider thích hợp
        providers = ort.get_available_providers()
        trt_provider_name = next((p for p in providers if 'Tensorrt' in p or 'TensorRT' in p), None)

        print(f"🚀 [JetSeg] Loading model...")
        
        if trt_provider_name:
            print(f"✅ Backend: TensorRT (FP16={use_fp16})")
            # Kiểm tra xem đã có file cache chưa để báo người dùng
            # Tên file cache của TRT thường rất dài và hash, nhưng ta chỉ cần check thư mục có file không
            if not os.listdir(cache_dir):
                print("⏳ LƯU Ý: Đây là lần chạy đầu tiên (hoặc vừa xóa cache).")
                print("   Hệ thống đang build TensorRT Engine (Mất ~1-2 phút). Vui lòng đợi...")
            
            self.session = ort.InferenceSession(self.model_path, providers=[(trt_provider_name, trt_options)], sess_options=sess_options)
        elif 'CUDAExecutionProvider' in providers:
            print("⚠️ Backend: CUDA (Chưa tối ưu bằng TensorRT)")
            self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'], sess_options=sess_options)
        else:
            print("⚠️ Backend: CPU (Rất chậm)")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)

        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image, threshold=0.5):
        if image is None: return None
        h_orig, w_orig = image.shape[:2]
        
        img_resized = cv2.resize(image, self.input_size)
        img_norm = img_resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_norm, axis=0)
        
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        pred_mask = outputs[0][0]
        pred_mask = cv2.resize(pred_mask, (w_orig, h_orig))
        
        return (pred_mask > threshold).astype(np.uint8) * 255

    def remove_background(self, image, mask, bg_color=(0, 255, 0)):
        green_bg = np.zeros_like(image)
        green_bg[:] = bg_color
        mask_3ch = np.expand_dims(mask > 0, axis=-1)
        return np.where(mask_3ch, image, green_bg)