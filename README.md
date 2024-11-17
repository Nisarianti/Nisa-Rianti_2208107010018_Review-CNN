![1](https://github.com/user-attachments/assets/8a80cd07-c033-45fc-b073-427ee22e18df)# Nisa-Rianti_2208107010018_Review-CNN
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
        
![1](https://github.com/user-attachments/assets/affdd6b3-84f0-4c58-b5e8-ad99f6ea0162)
1. Model CNN: Menggunakan model Sequential untuk membangun CNN dengan beberapa lapisan:
- Conv2D: Lapisan konvolusional untuk ekstraksi fitur dari gambar.
- MaxPooling2D: Pooling untuk mengurangi dimensi gambar.
- Flatten: Mengubah output 2D menjadi vektor 1D untuk lapisan Dense.
- Dense: Lapisan fully connected untuk klasifikasi.
- Augmentasi Data: Menggunakan ImageDataGenerator untuk memperbesar variasi data pelatihan melalui transformasi acak (flip, shear, zoom).
- Pelatihan: Model dilatih dengan fit_generator, menggunakan gambar dari direktori dan dioptimasi dengan algoritma Adam.


