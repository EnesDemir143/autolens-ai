# AutoLens AI — Kalibrasyon

Modelin verdiği olasılıkların gerçek doğruluk oranlarıyla uyumlu olması için post-hoc kalibrasyon uygulanmıştır. Kalibrasyon yalnızca **validation split** üzerinde fit edilmiş, **internal_test** üzerinde değerlendirilmiştir. Böylece test verisi model/kalibrasyon seçimi için kullanılmamıştır.

## Denenen Yöntemler

### Temperature Scaling
Logit değerleri tek bir sıcaklık parametresiyle ölçeklenir:

```text
softmax(logits / T)
```

Basit, hızlı ve prediction sıralamasını değiştirmez; ancak sınıf bazlı hataları sınırlı düzeltir.

### Vector Scaling
Her sınıf için ayrı ağırlık ve bias öğrenir:

```text
logits' = w * logits + b
```

Sınıf bazlı kalibrasyon hatalarını daha iyi düzeltebilir.

### Dirichlet Calibration
Softmax sonrası log-olasılık uzayında matris dönüşümü öğrenir:

```text
softmax(W · log(softmax(logits)) + b)
```

En esnek yöntemdir ve final modelde en iyi sonucu vermiştir.

## Sonuçlar

| Yöntem | ECE Önce | ECE Sonra | NLL Önce | NLL Sonra |
|---|---:|---:|---:|---:|
| Temperature Scaling | 0.0415 | 0.0136 | 0.2372 | 0.1784 |
| Vector Scaling | 0.0415 | 0.0121 | 0.2372 | 0.1768 |
| Dirichlet Calibration | 0.0984 | **0.0097** | 0.2644 | **0.1487** |

## Final Seçim

Seçilen yöntem: **Dirichlet Calibration (ODIR λ=0.001)**

Nedenleri:
- ECE değerini %1'in altına indiriyor.
- NLL değerinde en yüksek iyileştirmeyi sağlıyor.
- Modelin güven skorlarını daha yorumlanabilir hale getiriyor.

## Kullanım

`ONNXPredictor`, `artifacts/demo/active_model.json` içindeki `calibration_path` ve `calibration_method` alanlarını okuyarak kalibrasyonu otomatik uygular. Arayüz kodunda ekstra değişiklik gerekmez.

## Not

Kalibrasyon olasılık kalitesini artırır; sınıf sıralamasını/prediction argmax sonucunu zorunlu olarak değiştirmez. Bu yüzden accuracy/F1 yerine ECE ve NLL üzerinden değerlendirilmiştir.