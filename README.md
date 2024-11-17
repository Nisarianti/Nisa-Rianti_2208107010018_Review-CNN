# Nisa-Rianti_2208107010018_Review-CNN
# LINK 1 Deep Learning: Convolutional Neural Networks
    Artikel "Deep Learning: Convolutional Neural Networks" di Megabagus.id memperkenalkan konsep dasar CNN sebagai  teknik pembelajaran mendalam untuk mengolah data gambar. CNN menggunakan lapisan konvolusional untuk menangkap pola dalam gambar, seperti tepi, sudut, dan bentuk kompleks.Artikel ini juga menunjukkan cara menerapkan CNN pada masalah klasifikasi sederhana, seperti membedakan gambar kucing dan anjing.
    Aplikasi CNN di dunia nyata yang bisa kita lihat adalah Facebook. Dulu Anda harus menandai wajah orang, namun sekarang Facebook terkadang dapat secara otomatis menandai wajah teman dan keluarga Anda sebelum Anda menandai mereka. Ini adalah penerapan CNN di kehidupan nyata  di media sosial. Contoh lainnya adalah aplikasi CCTV (biasanya kamera kecil yang  dipasang di atap atau dinding) yang dapat mendeteksi wajah secara online secara real time. Di negara-negara maju, banyak sistem pengawasan video kini dipasang di sudut-sudut jalan. Selain untuk keamanan, kamera ini juga bisa digunakan untuk mengidentifikasi orang yang lalu lalang di depan Anda.
    Perbedaan antara gambar hitam putih dan berwarna dari sudut pandang komputer, serta bagaimana komputer menerjemahkan gambar menjadi angka melalui pengkodean pixel. Dengan menggunakan contoh sederhana seperti gambar 3×3 pixel, artikel ini berhasil menyederhanakan konsep yang kompleks sehingga mudah dipahami oleh pembaca umum. Penjelasannya terstruktur, mulai dari representasi pixel untuk gambar grayscale hingga array RGB untuk gambar berwarna, dan dilengkapi ilustrasi yang mendukung, seperti gambar wajah senyum dan kucing. Namun, terdapat kekeliruan dalam penjelasan grayscale yang menyebut angka desimal (0.3) untuk menggambarkan warna abu-abu, padahal komputer biasanya menggunakan nilai integer 0-255. Selain itu, pembahasan mengenai RGB kurang konkret karena tidak disertai contoh angka spesifik untuk warna tertentu. Artikel ini juga mengaitkan dasar-dasar pengolahan gambar dengan aplikasi machine learning dan deep learning, namun transisinya terasa kurang mendalam. Secara keseluruhan, artikel ini sangat informatif bagi pemula meski membutuhkan beberapa perbaikan dalam akurasi teknis dan kedalaman penjelasan.
Tahapan Convolutional Neural Network: 
- Convolution : Mengekstraksi fitur-fitur penting dari gambar (misalnya, tepi, pola, tekstur).
- Max Pooling : Mengurangi dimensi peta fitur sambil mempertahankan fitur yang paling penting, meningkatkan efisiensi komputasi, dan mengurangi risiko overfitting.
- Flattening : Mengubah peta fitur 2D (atau 3D untuk gambar berwarna) menjadi vektor 1D agar dapat diproses oleh lapisan jaringan saraf biasa (fully connected layer).
- Full Connection : Melakukan klasifikasi berdasarkan fitur-fitur yang telah diekstraksi.
    Proses CNN dimulai dengan Convolution untuk mengekstraksi fitur, Max Pooling untuk mengurangi dimensi, Flattening untuk mempersiapkan data dalam format vektor, dan Full Connection untuk menghasilkan prediksi akhir. Kombinasi tahapan ini memungkinkan CNN menjadi alat yang sangat efektif dalam tugas-tugas seperti klasifikasi gambar, deteksi objek, dan segmentasi gambar.


# LINK 2 Deep Learning: Convolutional Neural Networks (aplikasi)

        Convolutional Neural Networks (CNN) adalah salah satu jenis arsitektur deep learning yang sangat efektif dalam menangani tugas-tugas terkait pengolahan gambar. CNN dirancang untuk mengidentifikasi pola visual di dalam gambar, seperti tepi, tekstur, dan bentuk. Teknik ini menggunakan lapisan konvolusi untuk memindai gambar dengan filter tertentu, yang membantu mengekstrak fitur penting, diikuti dengan lapisan pooling untuk mereduksi dimensi dan menangkap informasi utama. CNN sangat populer dalam aplikasi pengenalan gambar, klasifikasi objek, dan deteksi wajah. Keunggulan utama dari CNN adalah kemampuannya untuk belajar dari data gambar secara otomatis, tanpa memerlukan fitur yang diekstraksi secara manual. Dalam penerapannya, dataset gambar dibagi menjadi set pelatihan dan pengujian, dan CNN dilatih menggunakan gambar-gambar tersebut untuk membedakan objek dalam gambar, seperti kucing dan anjing. Dengan menggunakan library Keras di Python, kita dapat membangun dan melatih model CNN untuk tugas klasifikasi gambar dengan cara yang cukup efisien.
# Code 1
![1](https://github.com/user-attachments/assets/affdd6b3-84f0-4c58-b5e8-ad99f6ea0162)

penjelasan kode:
# Mengimpor Library yang Diperlukan
![image](https://github.com/user-attachments/assets/1b23bd2c-9f0c-4d4d-81d1-df79593c373a)
- Sequential: Model bertipe sequential digunakan untuk membuat jaringan neural dengan lapisan yang disusun secara linear (berturut-turut).
- Conv2D: Lapisan Convolutional digunakan untuk ekstraksi fitur dari gambar.
- MaxPooling2D: Lapisan MaxPooling digunakan untuk mengurangi dimensi gambar setelah konvolusi, yang membantu dalam mengurangi jumlah parameter.
- Flatten: Digunakan untuk mengubah data 2D menjadi 1D sehingga dapat diproses oleh lapisan Dense.
- Dense: Lapisan fully connected yang digunakan untuk klasifikasi akhir.
  
# Inisialisasi CNN
![image](https://github.com/user-attachments/assets/749309e5-8e12-4d08-8398-5b559d3eba2b)
- Inisialisasi model CNN menggunakan Sequential.

# langkah-langkah
![image](https://github.com/user-attachments/assets/e228582b-8cd7-4324-8434-e83f3cef7097)
- Conv2D: Lapisan konvolusional dengan 32 filter, ukuran kernel 3x3.
- input_shape=(128, 128, 3): Ukuran gambar input adalah 128x128 piksel dengan 3 saluran (RGB).
- activation='relu': Menggunakan ReLU sebagai fungsi aktivasi untuk lapisan ini.
- MaxPooling2D: Melakukan max pooling dengan ukuran pool 2x2. Ini mengurangi ukuran gambar dan mengurangi kompleksitas komputasi.
- Menambahkan satu lapisan konvolusional lagi dengan 32 filter dan ukuran kernel 3x3, diikuti dengan max pooling.
- Flatten: Mengubah output 2D dari lapisan sebelumnya menjadi vektor 1D agar bisa diproses oleh lapisan Dense.
- Lapisan pertama Dense memiliki 128 unit dan fungsi aktivasi ReLU.
- Lapisan terakhir Dense memiliki 1 unit dan fungsi aktivasi sigmoid, digunakan untuk output biner (klasifikasi dua kelas).

# Menjalankan CNN
![image](https://github.com/user-attachments/assets/5d44e0b3-fbf1-4ec7-95c3-d302b5d995cd)
- compile: Menyusun model untuk pelatihan.
- optimizer='adam': Menggunakan Adam sebagai algoritma optimasi.
- loss='binary_crossentropy': Fungsi kerugian yang digunakan untuk klasifikasi biner.
- metrics=['accuracy']: Menggunakan akurasi sebagai metrik evaluasi.

# Menyiapkan Data Augmentasi dan Memuat Dataset
![image](https://github.com/user-attachments/assets/35403841-b0de-491e-832b-5d6084f0928b)
- ImageDataGenerator: Digunakan untuk augmentasi gambar yang membantu memperbaiki generalisasi model.
  - rescale = 1./255: Menormalkan gambar ke rentang 0-1.
  - shear_range, zoom_range, horizontal_flip: Mengaplikasikan transformasi acak pada gambar pelatihan untuk meningkatkan variasi data.
- flow_from_directory: Memuat gambar dari direktori, mengubah ukuran gambar menjadi 128x128 dan mengonversinya ke format batch untuk pelatihan.

# output yang dihasilkan 
![WhatsApp Image 2024-11-17 at 20 43 08_d82d76f6](https://github.com/user-attachments/assets/b3db33c8-b9af-4b4f-9328-9e4d58e90b6b)
Tampilan eksekusi epoch ke 50 di spyder
Kita bisa melihat bahwa nilai loss di training set menjadi sangat kecil yaitu 0.000. Walaupun nilainya sudah 0, namun dari iterasi epoch di atasnya nilai ini masih bisa terus turun (jadi bisa ditambah lagi jumlah epoch-nya).
Kemudian nilai akurasinya sudah sangat tinggi yaitu 93%. Nilai ini juga masih bisa ditingkatkan lagi dengan menambah epoch nya.
Val_loss menunjukkan angka yang juga sangat kecil yaitu 0.6397.


# Code 2
![image](https://github.com/user-attachments/assets/f0787675-7b24-48e1-a5cc-3b6f9a428351)



# LINK 2 Deep Learning: Convolutional Neural Networks (aplikasi)


