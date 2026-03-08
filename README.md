# Sınav Linear Regression Modeli

Bu proje, PyTorch kullanarak basit bir lineer regresyon modeli ile sınav notlarını çalışma süresine göre tahmin eder.

## Özellikler

- **Veri Seti**: Öğrencilerin çalışma saatleri ve aldıkları notlar
- **Model**: Tek katmanlı lineer regresyon
- **Optimizasyon**: Stochastic Gradient Descent (SGD)
- **Performans Metrikleri**: R² Score ve Mean Squared Error (MSE)

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install torch pandas matplotlib scikit-learn
```

2. Veri dosyasını (`06-study_hours_grades.csv`) aynı dizine koyun.

3. Kodu çalıştırın:
```bash
python d1.py
```

## Çıktılar

- Eğitim süreci sırasında loss değerleri
- Final model performans skorları (R² ve MSE)
- Eğitim/test verisi ve tahminlerin görselleştirilmesi

## Model Performansı

Mevcut model yaklaşık **%92 R² skoru** ile oldukça başarılı sonuçlar vermektedir.

## Kullanılan Teknolojiler

- PyTorch
- Pandas
- Matplotlib
- Scikit-learn