import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# --- Model dan Label ---
try:
    model = keras.models.load_model("mosquito_v1.h5")
    # Tambahkan log untuk memastikan model termuat dengan benar
    print("Model mosquito_v1.h5 berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model: {e}")
    # Anda mungkin ingin keluar atau memberi tahu admin jika model tidak bisa dimuat
    exit(1)

label = [
    'Aedes Aegypti', 'Aedes Albopictus', 'Anopheles Albimanus', 'Anopheles Arabiensis',
    'Anopheles Atroparvus', 'Anopheles Coluzzi', 'Anopheles Farauti',
    'Anopheles Freeborni', 'Anopheles Stephensi', 'Culex Quinquefasciatus', 'Unknown'
]

keterangan_label = {
    'Aedes Aegypti': 'Ciri fisik utama adalah corak belang (loreng) putih dan hitam pada kaki dan tubuhnya. Di bagian punggung (dorsal) terdapat pola khas berbentuk seperti alat musik lira (lyre) berwarna putih.',
    'Aedes Albopictus': 'Dikenal sebagai nyamuk macan, memiliki tubuh berwarna hitam dengan satu garis putih tebal dan jelas tepat di tengah punggungnya. Kakinya juga memiliki belang-belang putih, namun lebih sedikit dibanding Aedes aegypti.',
    'Anopheles Albimanus': 'Memiliki tubuh berwarna gelap. Ciri khas utamanya adalah sisik putih pada bagian tarsus (segmen ujung) kaki belakangnya. Saat istirahat, posisi tubuhnya menungging khas nyamuk Anopheles.',
    'Anopheles Arabiensis': 'Spesies ini sangat mirip dengan Anopheles gambiae dan Anopheles coluzzi. Identifikasi visual sulit dan seringkali membutuhkan analisis genetik atau morfologi detail. Umumnya berwarna coklat keabu-abuan dengan bintik-bintik samar di sayap.', # **TAMBAHAN/PERBAIKAN INI**
    'Anopheles Atroparvus': 'Nyamuk berukuran sedang dengan warna coklat kusam. Sayapnya memiliki bintik-bintik gelap yang tidak terlalu kontras. Sulit dibedakan secara visual dari Anopheles lain tanpa mikroskop.',
    'Anopheles Coluzzi': 'Secara fisik hampir identik dengan Anopheles gambiae. Nyamuk ini berwarna coklat muda hingga abu-abu dengan bintik-bintik gelap yang tersebar di sayapnya. Identifikasi pasti memerlukan analisis genetik.',
    'Anopheles Farauti': 'Memiliki tubuh berwarna gelap. Sisik pada sayapnya membentuk pola bintik-bintik gelap dan pucat yang jelas. Ujung sayap (wing tip) seringkali memiliki pinggiran pucat.',
    'Anopheles Freeborni': 'Berwarna coklat keabu-abuan dengan bintik-bintik gelap pada sayapnya. Ciri khasnya adalah adanya kumpulan sisik gelap pada vena sayap tertentu. Betina memiliki palpi (organ dekat mulut) sepanjang proboscis (belalai).',
    'Anopheles Stephensi': 'Memiliki bintik-bintik pada tubuh dan kaki. Ciri khasnya adalah adanya belang putih dan hitam pada tarsus (ujung kaki) dan proboscis (belalai) yang berbintik.',
    'Culex Quinquefasciatus': 'Umumnya berwarna coklat muda tanpa pola atau corak yang mencolok pada tubuh dan kakinya. Saat istirahat, posisi tubuhnya sejajar dengan permukaan (tidak menungging). Abdomen (perut) memiliki ujung yang tumpul.',
}

app = Flask(__name__)

def predict_label(img):
    i = np.asarray(img) / 255.0
    # Pastikan gambar memiliki 3 channel (RGB)
    if i.shape[-1] == 4: # Jika gambar RGBA, konversi ke RGB
        i = i[..., :3]
    elif i.shape[-1] == 1: # Jika gambar Grayscale, konversi ke RGB dengan duplikasi channel
        i = np.stack([i[:, :, 0]] * 3, axis=-1)

    i = i.reshape(1, 224, 224, 3) # Pastikan bentuk input sesuai model

    pred = model.predict(i)

    # Tambahkan log untuk melihat output prediksi mentah
    print(f"Prediksi mentah dari model: {pred}")
    print(f"Bentuk prediksi mentah: {pred.shape}")

    # Validasi bentuk output prediksi
    if pred.shape[1] != len(label):
        raise ValueError(
            f"Bentuk output model {pred.shape[1]} tidak sesuai dengan jumlah label {len(label)}. "
            "Pastikan model melatih dengan jumlah kelas yang sama dengan label yang diberikan."
        )

    class_index = np.argmax(pred)
    confidence = float(np.max(pred))
    result = label[class_index]
    return result, confidence

@app.route("/predict", methods=["POST"])
def index():
    file = request.files.get('image')
    if file is None or file.filename == "":
        return jsonify({"error": "no file image"}), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Menggunakan Image.LANCZOS untuk downsampling berkualitas tinggi (lebih baik dari NEAREST)
        # NEAREST bisa menyebabkan aliasing dan kualitas buruk pada gambar yang di-resize
        img = img.resize((224, 224), Image.LANCZOS)

        # Pastikan gambar diubah ke mode RGB sebelum diolah oleh model
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pred_img, confidence = predict_label(img)

        response_data = {}
        if pred_img == 'Unknown':
            response_data['hasil'] = "Gambar tidak terdeteksi"
            response_data['keterangan'] = "Gambar yang Anda masukkan tidak dapat diidentifikasi sebagai salah satu jenis nyamuk dalam database kami. Silakan coba lagi dengan gambar yang lebih jelas."
        else:
            response_data['hasil'] = pred_img
            response_data['keterangan'] = keterangan_label.get(pred_img, "Keterangan fisik tidak tersedia.")

        response_data['akurasi'] = f"{confidence * 100:.2f}%"

        return jsonify(response_data), 200

    except ValueError as ve:
        # Tangani ValueError secara spesifik (misalnya, jika bentuk model tidak sesuai)
        print(f"Error konfigurasi model: {ve}")
        return jsonify({"error": "Kesalahan konfigurasi model", "detail": str(ve)}), 500
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "gagal memproses gambar", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)