import re
import numpy as np

# Fungsi untuk menghitung jarak antara dua vektor
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

# Fungsi untuk mengonversi teks menjadi vektor BoW
def text_to_bow(text, vocabulary):
    words = re.findall(r'\w+', text.lower())
    vector = [words.count(word) for word in vocabulary]
    return np.array(vector)

# Fungsi untuk menentukan label kelas dari dokumen uji
def classify(train_data, train_labels, test_data, k):
    num_train = len(train_data)
    num_test = len(test_data)
    predictions = []

    for i in range(num_test):
        distances = [euclidean_distance(test_data[i], train_data[j]) for j in range(num_train)]
        nearest_neighbors = np.argsort(distances)[:k]
        nearest_labels = [train_labels[idx] for idx in nearest_neighbors]
        predicted_label = max(set(nearest_labels), key=nearest_labels.count)
        predictions.append(predicted_label)

    return predictions

# Data latihan dan data uji (contoh)
train_texts = [
    "Berita olahraga tentang pertandingan sepak bola",
    "Politik hari ini membahas kebijakan pemerintah",
    "Terkini: Teknologi mutakhir dalam dunia komputasi",
    "dunia politik sedang memanas"
]
train_labels = ["Olahraga", "Politik", "Teknologi", "Politik"]

test_texts = [
    "Olahraga pagi ini",
]
k = 1

# Membangun vektor fitur BoW untuk data latihan dan data uji
vocabulary = set(re.findall(r'\w+', ' '.join(train_texts + test_texts).lower()))
train_data = np.array([text_to_bow(text, vocabulary) for text in train_texts])
test_data = np.array([text_to_bow(text, vocabulary) for text in test_texts])

# Klasifikasi dokumen
predictions = classify(train_data, train_labels, test_data, k)

# Menampilkan hasil klasifikasi
for i in range(len(predictions)):
    print(f"Data uji {i + 1} diklasifikasikan sebagai: {predictions[i]}")
