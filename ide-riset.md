Berikut **dua topik penelitian** yang menurut saya paling tepat dari data Anda. Saya buatkan juga **intro/pendahuluan awal** yang bisa langsung dikembangkan menjadi Bab 1 atau artikel jurnal.

---

# **Topik 1 — Fokus Kebijakan Publik**

## **Judul Indonesia**

**Analisis Sentimen Berbasis Aspek dan Emosi Publik terhadap Pelaksanaan Program Makan Bergizi Gratis pada Media Sosial Threads**

## **Judul English**

**Aspect-Based Sentiment and Emotion Analysis of Public Perception toward the Implementation of Indonesia’s Free Nutritious Meal Program on Threads**

## **Arah penelitian**

Topik ini fokus pada **apa yang dirasakan masyarakat terhadap pelaksanaan MBG**, lalu dihubungkan dengan aspek-aspek implementasi seperti:

| Aspek            | Contoh isu                                     |
| ---------------- | ---------------------------------------------- |
| Keamanan pangan  | keracunan, makanan basi, higienitas, SOP dapur |
| Kualitas menu    | rasa, porsi, gizi, variasi makanan             |
| Distribusi       | keterlambatan, jangkauan, teknis pengiriman    |
| Sasaran penerima | siswa, ibu hamil, balita, tepat sasaran        |
| Anggaran         | efisiensi, transparansi, dugaan pemborosan     |
| Tata kelola SPPG | dapur, mitra, kapasitas produksi, pengawasan   |
| Dampak sosial    | UMKM, petani, tenaga kerja lokal               |

## **Intro / Pendahuluan**

Program Makan Bergizi Gratis atau MBG merupakan salah satu kebijakan strategis nasional yang bertujuan meningkatkan kualitas sumber daya manusia melalui pemenuhan gizi bagi kelompok penerima manfaat. Pada tahun 2026, Badan Gizi Nasional menargetkan program ini dapat menjangkau **82,9 juta penerima manfaat**, dengan proyeksi penyediaan sekitar **21 miliar porsi makanan** dan perluasan ribuan Satuan Pelayanan Pemenuhan Gizi atau SPPG di berbagai wilayah Indonesia. Skala program yang sangat besar tersebut menjadikan MBG bukan hanya program bantuan pangan, tetapi juga kebijakan publik nasional yang memiliki dampak sosial, ekonomi, kesehatan, dan politik yang luas. ([Badan Gizi Nasional][1])

Di tengah perluasan implementasinya, MBG memunculkan respons publik yang beragam. Pemerintah menekankan pentingnya penguatan tata kelola, pengawasan keamanan pangan, validasi data penerima manfaat, serta standar mutu layanan agar program berjalan aman dan tepat sasaran. Kementerian Kesehatan juga menyatakan bahwa keselamatan anak menjadi prioritas dalam penguatan tata kelola MBG, terutama setelah munculnya sejumlah kejadian yang mendorong evaluasi pelaksanaan program. ([Kementerian Kesehatan RI][2])

Media sosial menjadi ruang penting bagi masyarakat untuk menyampaikan dukungan, kritik, kekhawatiran, dan pengalaman mereka terhadap pelaksanaan MBG. Berbeda dari survei formal, percakapan media sosial dapat menangkap respons spontan masyarakat secara lebih luas dan real-time. Dalam konteks ini, Threads menjadi sumber data yang menarik karena digunakan sebagai ruang diskusi sosial yang lebih naratif, personal, dan percakapan dibandingkan platform lain. Data yang telah dikumpulkan dalam penelitian ini menunjukkan bahwa sentimen negatif menjadi kategori terbesar, yaitu **2.990 data**, disusul netral **2.354 data**, positif **1.130 data**, dan mixed **541 data**. Selain itu, emosi yang dominan adalah **neutral**, **frustration**, dan **anger**, yang menunjukkan bahwa opini publik terhadap MBG tidak cukup dianalisis hanya dalam kategori positif, negatif, dan netral.

Penelitian terdahulu mengenai sentimen MBG sudah banyak dilakukan pada platform seperti Twitter/X, Instagram, YouTube, dan TikTok, dengan pendekatan seperti SVM, LSTM, IndoBERT, serta analisis sentimen berbasis aspek. Namun, sebagian besar studi masih berfokus pada klasifikasi sentimen umum, belum banyak yang menggabungkan **deteksi relevansi, klasifikasi emosi, dan analisis berbasis aspek** secara terpadu, khususnya menggunakan data Threads. Studi YouTube MBG misalnya menggunakan LSTM terhadap 7.733 komentar, sedangkan studi lain menganalisis komentar Instagram, Twitter/X, dan TikTok. ([arXiv][3])

Berdasarkan kondisi tersebut, penelitian ini bertujuan untuk menganalisis persepsi publik terhadap pelaksanaan Program MBG melalui pendekatan **aspect-based sentiment analysis** dan **emotion analysis**. Dengan mengidentifikasi aspek pelaksanaan yang paling banyak memicu sentimen negatif, positif, mixed, atau netral, penelitian ini diharapkan dapat memberikan kontribusi praktis bagi evaluasi komunikasi kebijakan publik serta kontribusi akademik dalam pengembangan analisis opini digital berbasis emosi dan aspek.

## **Novelty**

Novelty-nya kuat di bagian ini:

> Penelitian ini tidak hanya mengklasifikasikan sentimen publik terhadap MBG, tetapi juga menggabungkan deteksi relevansi, klasifikasi emosi, dan analisis berbasis aspek untuk mengidentifikasi isu implementasi yang paling banyak memicu frustrasi, kemarahan, kekhawatiran, kepercayaan, dan kepuasan publik pada media sosial Threads.

## **Rumusan masalah**

1. Bagaimana distribusi sentimen publik terhadap pelaksanaan Program MBG di Threads?
2. Emosi apa yang paling dominan dalam percakapan publik terkait MBG?
3. Aspek implementasi MBG apa yang paling banyak memicu sentimen negatif, positif, netral, dan mixed?
4. Bagaimana hubungan antara aspek implementasi dengan emosi publik seperti frustration, anger, worry, trust, dan satisfaction?
5. Bagaimana hasil analisis sentimen dan emosi dapat digunakan sebagai masukan evaluasi kebijakan publik?

---

# **Topik 2 — Fokus Teknologi / Model NLP**

## **Judul Indonesia**

**Pengembangan Model Deteksi Relevansi, Sentimen, dan Emosi pada Percakapan Program MBG di Threads Menggunakan Pendekatan Machine Learning dan Large Language Model**

## **Judul English**

**Developing a Relevance-Aware Sentiment and Emotion Classification Model for Public Discourse on the Free Nutritious Meal Program Using Threads Data**

## **Arah penelitian**

Topik kedua lebih teknis. Fokusnya bukan hanya membaca opini publik, tetapi **membangun model klasifikasi teks** yang mampu:

1. Menentukan apakah teks relevan dengan topik MBG.
2. Mengklasifikasikan sentimen: positive, negative, neutral, mixed.
3. Mengklasifikasikan emosi: neutral, frustration, anger, joy, trust, confusion, satisfaction, disappointment, worry, surprise.
4. Membandingkan performa beberapa model NLP.
5. Menangani data tidak seimbang.

Ini cocok kalau target Anda masuk ke jurnal bidang:

* data science
* artificial intelligence
* NLP
* information system
* social media analytics
* machine learning untuk kebijakan publik

## **Intro / Pendahuluan**

Perkembangan media sosial telah menghasilkan volume data opini publik yang sangat besar dan dinamis. Dalam isu kebijakan publik, data media sosial dapat digunakan untuk memahami bagaimana masyarakat merespons program pemerintah, termasuk dalam bentuk dukungan, kritik, kekhawatiran, kebingungan, maupun ketidakpuasan. Program Makan Bergizi Gratis atau MBG merupakan salah satu isu publik yang banyak diperbincangkan karena memiliki skala implementasi nasional dan menyasar puluhan juta penerima manfaat. Pada awal 2026, BGN menyatakan bahwa program MBG telah menjangkau hampir 60 juta penerima manfaat dan terus diarahkan menuju target 82,9 juta penerima manfaat. ([Badan Gizi Nasional][4])

Dalam praktiknya, analisis opini publik terhadap MBG menghadapi beberapa tantangan. Pertama, tidak semua percakapan yang mengandung kata kunci MBG benar-benar relevan dengan pelaksanaan program. Data penelitian ini menunjukkan terdapat **2.039 data tidak relevan** dan **1 data None**, sehingga proses deteksi relevansi menjadi penting sebelum analisis sentimen dilakukan. Kedua, distribusi sentimen tidak seimbang, dengan kelas negatif lebih dominan dibandingkan positif dan mixed. Ketiga, opini publik tidak selalu dapat direduksi menjadi positif, negatif, atau netral karena terdapat dimensi emosi seperti frustration, anger, worry, confusion, trust, dan satisfaction.

Sebagian penelitian terdahulu telah menggunakan berbagai algoritma machine learning dan deep learning untuk menganalisis sentimen MBG. Studi pada YouTube menggunakan LSTM dan memperoleh akurasi 89%, tetapi juga menunjukkan tantangan ketidakseimbangan kelas karena performa terhadap sentimen positif lebih rendah dibandingkan sentimen negatif. Studi lain menggunakan SVM pada data TikTok, analisis komentar Instagram, serta IndoBERT dan DistilBERT pada data Twitter/X. Hal ini menunjukkan bahwa analisis sentimen MBG mulai berkembang, tetapi masih terdapat ruang penelitian untuk mengembangkan model yang lebih komprehensif dengan memasukkan **relevance detection** dan **emotion classification**. ([arXiv][3])

Berdasarkan hal tersebut, penelitian ini bertujuan untuk mengembangkan model klasifikasi teks berbasis machine learning dan/atau large language model untuk mendeteksi relevansi, sentimen, dan emosi publik terhadap Program MBG pada media sosial Threads. Model yang dikembangkan diharapkan tidak hanya mampu mengklasifikasikan opini menjadi positif, negatif, netral, dan mixed, tetapi juga mampu menangkap emosi yang lebih spesifik sehingga hasil analisis menjadi lebih informatif bagi evaluasi kebijakan publik.

Secara teknis, penelitian ini dapat membandingkan beberapa pendekatan, seperti TF-IDF dengan SVM sebagai baseline, IndoBERT atau IndoBERTweet sebagai model deep learning utama, serta pendekatan LLM zero-shot atau few-shot sebagai pembanding. Evaluasi model dilakukan menggunakan precision, recall, F1-score, macro-F1, weighted-F1, dan confusion matrix, terutama karena data memiliki distribusi kelas yang tidak seimbang. Dengan demikian, penelitian ini tidak hanya menghasilkan pemetaan opini publik terhadap MBG, tetapi juga memberikan kontribusi pada pengembangan model NLP berbahasa Indonesia untuk isu kebijakan publik.

## **Novelty**

Novelty-nya bisa ditulis seperti ini:

> Penelitian ini mengembangkan pendekatan klasifikasi bertingkat yang menggabungkan deteksi relevansi, klasifikasi sentimen, dan klasifikasi emosi pada data Threads terkait Program MBG. Berbeda dari penelitian sebelumnya yang umumnya hanya mengklasifikasikan sentimen, penelitian ini menambahkan lapisan relevance-aware dan emotion-aware classification untuk meningkatkan kualitas analisis opini publik.

## **Rumusan masalah**

1. Bagaimana membangun model deteksi relevansi untuk menyaring percakapan Threads yang benar-benar berkaitan dengan Program MBG?
2. Bagaimana performa model machine learning dan deep learning dalam mengklasifikasikan sentimen publik terhadap MBG?
3. Bagaimana performa model dalam mengklasifikasikan emosi publik seperti frustration, anger, trust, worry, dan satisfaction?
4. Model mana yang memberikan performa terbaik berdasarkan macro-F1 dan weighted-F1?
5. Bagaimana pengaruh ketidakseimbangan data terhadap performa klasifikasi sentimen dan emosi?
