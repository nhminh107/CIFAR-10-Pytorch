# CIFAR-10 Image Classification

Project xây dựng mô hình Convolutional Neural Network (CNN) bằng PyTorch để phân loại ảnh trong bộ dữ liệu CIFAR-10.

Mỗi ảnh có kích thước `32x32` pixel và thuộc một trong 10 lớp:

- `airplane`
- `automobile`
- `bird`
- `cat`
- `deer`
- `dog`
- `frog`
- `horse`
- `ship`
- `truck`

Cuộc thi Kaggle: [CIFAR-10 Object Recognition in Images](https://www.kaggle.com/competitions/cifar-10)

## Kết quả

Kaggle score: **0.75740**

## Cấu trúc project

```text
CIFAR-10_Project/
├── data/
│   ├── train/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── test/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── trainLabels.csv
├── CONFIG.py
├── data.py
├── model.py
├── train.py
├── test.py
├── best_model.pth
└── submission.csv
```

### Các file chính

- `CONFIG.py`: chứa device, batch size, danh sách lớp và ánh xạ label.
- `data.py`: định nghĩa custom PyTorch `Dataset` để đọc tập train và test.
- `model.py`: định nghĩa kiến trúc CNN dùng để phân loại ảnh.
- `train.py`: huấn luyện, validation, early stopping và lưu model tốt nhất.
- `test.py`: nạp `best_model.pth`, dự đoán tập test và tạo file submission.
- `best_model.pth`: trọng số model có validation loss tốt nhất.
- `submission.csv`: kết quả dự đoán theo định dạng Kaggle `id,label`.

## Dataset

Dataset được tải từ cuộc thi Kaggle và đặt trong thư mục `data`.

### Tập train

- Ảnh train nằm trong `data/train`.
- Tên file ảnh là ID, ví dụ `1.png`.
- Label tương ứng được lưu trong `data/trainLabels.csv` với hai cột:

```csv
id,label
1,frog
2,truck
```

### Tập test

- Ảnh test nằm trong `data/test`.
- Tập test không có label.
- `test.py` sử dụng ID từ tên file ảnh để tạo kết quả dự đoán.

## Mô hình

Mô hình CNN gồm ba khối convolution. Mỗi khối sử dụng các thành phần như:

- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d`
- `Dropout`

Phần classifier sử dụng hai lớp fully connected và trả về logits cho 10 lớp CIFAR-10.

Ảnh train được augmentation bằng random crop và horizontal flip. Cả tập train, validation và test đều được chuẩn hóa theo mean và standard deviation của CIFAR-10.

## Cài đặt

Các thư viện chính:

```bash
pip install torch torchvision pandas pillow
```

## Huấn luyện

```bash
python train.py
```

Trong quá trình huấn luyện, model có validation loss thấp nhất được lưu vào:

```text
best_model.pth
```

Quá trình train dừng sớm nếu validation loss không cải thiện sau 5 epoch liên tiếp.

## Dự đoán và tạo submission

```bash
python test.py
```

Kết quả được lưu tại `submission.csv` theo định dạng:

```csv
id,label
1,deer
2,airplane
3,automobile
```

File này có thể được upload trực tiếp lên trang cuộc thi Kaggle để chấm điểm.
