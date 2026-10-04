# tests/evaluate_local.py
"""
Evaluasi Lokal SiagaChat.
Menguji model terhadap URL buatan sendiri (modus lokal Indonesia).
"""
import os
import sys
import pandas as pd

# Pastikan path ke folder src benar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_url

def main():
    print("=" * 70)
    print("SIAGACHAT: EVALUASI LOKAL (INDONESIAN PHISHING SCENARIOS)")
    print("=" * 70)
    
    csv_path = os.path.join(os.path.dirname(__file__), 'local_urls.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: File {csv_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(csv_path)
    
    correct_count = 0
    false_positives = [] # Model bilang Bahaya, padahal Aman
    false_negatives = [] # Model bilang Aman/Waspada, padahal Bahaya
    
    print(f"\nMenguji {len(df)} URL...\n")
    print(f"{'STATUS':<6} | {'SKOR':<5} | {'PREDIKSI':<8} | {'EKSPETASI':<8} | URL")
    print("-" * 70)
    
    for _, row in df.iterrows():
        url = row['url'].strip() # Hapus spasi tersembunyi
        expected = int(row['expected_label'])
        
        try:
            result = predict_url(url)
            risk_score = result['risk_score']
            pred_label_str = result['label']
            
            # Konversi prediksi ke 0/1 untuk perbandingan
            # Aman = 0, Waspada/Bahaya = 1
            predicted = 0 if pred_label_str == "Aman" else 1
            
            is_correct = (predicted == expected)
            
            if is_correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"
                if expected == 0 and predicted == 1:
                    false_positives.append((url, risk_score, pred_label_str))
                else:
                    false_negatives.append((url, risk_score, pred_label_str))
            
            exp_str = "Aman (0)" if expected == 0 else "Bahaya (1)"
            print(f"{status:<6} | {risk_score:<5} | {pred_label_str:<8} | {exp_str:<8} | {url}")
            
        except Exception as e:
            print(f"❌ ERROR | -     | -        | -        | {url} -> {e}")

    # Ringkasan
    accuracy = (correct_count / len(df)) * 100
    
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL")
    print("=" * 70)
    print(f"Total URL Diuji   : {len(df)}")
    print(f"Prediksi Benar    : {correct_count}")
    print(f"Prediksi Salah    : {len(df) - correct_count}")
    print(f"Akurasi Lokal     : {accuracy:.1f}%")
    
    if false_negatives:
        print("\n⚠️  FALSE NEGATIVES (Bahaya tapi lolos/terdeteksi rendah):")
        print("   (Model menganggap ini aman/waspada, padahal ini link penipuan)")
        for url, score, label in false_negatives:
            print(f"   - [{label}] Skor: {score} | {url}")
            
    if false_positives:
        print("\n⚠️  FALSE POSITIVES (Aman tapi dituduh berbahaya):")
        print("   (Model menganggap ini berbahaya, padahal ini link resmi)")
        for url, score, label in false_positives:
            print(f"   - [{label}] Skor: {score} | {url}")
            
    if not false_negatives and not false_positives:
        print("\n🎉 SEMPURNA! Model tidak melakukan kesalahan klasifikasi sama sekali!")

if __name__ == "__main__":
    main()