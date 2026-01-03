# BERT-ROBERTA
Automatic Ticket Classification using BERT & RoBERTa

Bu repository, müşteri şikayetleri (complaints) üzerinde otomatik ticket / ürün sınıflandırması yapmak amacıyla geliştirilmiş iki farklı Transformer tabanlı derin öğrenme modelini içermektedir:

BERT (TensorFlow / Keras)

RoBERTa (PyTorch / Hugging Face)

Amaç, serbest metin şeklindeki müşteri şikayetlerini doğru ürün / kategori sınıflarına otomatik olarak atamaktır.

📌 Proje Amacı

Uzun ve serbest metinlerden oluşan müşteri şikayetlerini analiz etmek

Manuel ticket yönlendirme ihtiyacını azaltmak

Çağrı merkezi ve müşteri destek sistemlerinde otomasyonu artırmak

BERT ve RoBERTa modellerinin performanslarını karşılaştırmak

📂 Kullanılan Veri Seti

filtered_complaints.json

Her satır bir müşteri şikayetini temsil eder

Temel Sütunlar:

complaint_what_happened → Şikayet metni

product → Şikayetin ait olduğu ürün / kategori

🧠 Modeller ve Dosyalar
1️⃣ bertautomaticticketclassification.py

(TensorFlow + BERT)

bert-base-uncased modeli kullanılmıştır

TensorFlow TFAutoModelForSequenceClassification

Özellikler:

Train / Validation / Test split (%70 / %15 / %15)

Tokenization + padding

Class weight ile dengesiz veri problemi çözümü

Early Stopping & Model Checkpoint

Accuracy & Loss grafikleri

Classification Report (Precision, Recall, F1)

📌 Avantajı
TensorFlow ekosistemiyle kolay entegrasyon ve hızlı prototipleme

2️⃣ roberta.py

(PyTorch + RoBERTa)

roberta-base modeli kullanılmıştır

Hugging Face Transformers + PyTorch

Özellikler:

Label Encoding & mapping

Dengesiz sınıflar için weighted loss

Custom training loop

AMP (fp16) desteği

Early Stopping

Macro F1-score odaklı değerlendirme

Detaylı test raporları ve grafikler

📌 Avantajı
Daha güçlü dil temsili ve genellikle BERT’e göre daha yüksek performans

🧹 Veri Ön İşleme

Eksik değerlerin temizlenmesi

Metinlerin string’e dönüştürülmesi

Uzunluk sınırlandırma (MAX_LENGTH)

Stratified train / validation / test bölme

⚙️ Kullanılan Teknolojiler

Python

TensorFlow / Keras

PyTorch

Hugging Face Transformers

Hugging Face Datasets

Scikit-learn

NumPy, Pandas

Matplotlib

📊 Değerlendirme Metrikleri

Accuracy

Precision / Recall / F1-score

Macro F1 (özellikle dengesiz veri için)

Confusion Matrix (opsiyonel)

▶️ Çalıştırma
Ortam Kurulumu
pip install transformers datasets accelerate torch tensorflow scikit-learn pandas numpy matplotlib

BERT modeli için:
python bertautomaticticketclassification.py

RoBERTa modeli için:
python roberta.py


⚠️ GPU önerilir. Uzun metinler ve çok sınıflı yapı nedeniyle CPU’da eğitim yavaş olabilir.

🚀 Geliştirme Fikirleri

DistilBERT / RoBERTa-large denemeleri

Hiperparametre optimizasyonu

Model servisleştirme (FastAPI)

Gerçek zamanlı ticket sınıflandırma sistemi

Çok dilli destek

👩‍💻 Hazırlayan

Gizem Bektaş
Natural Language Processing & Deep Learning
