# Peta Dokumentasi — YOLO Bukukia Training Pipeline

Mulailah dari panduan dari nol jika baru mengenal pipeline ini. Proyek ini tidak memiliki aplikasi web; dokumentasinya berfokus pada data, training, evaluasi, dan penyerahan model.

> **Catatan keamanan (Agustus 2026)**: repo ini sebelumnya tidak punya `.gitignore` sama sekali, padahal `input_files/`, `input_files_kia2024/`, dan `results/` berisi PII (nama ibu/ayah, NIK anak) — termasuk di level nama file, bukan cuma isi gambar. `.gitignore` sudah ditambahkan di root untuk mengecualikan folder-folder ini dan `*.pt` (model weights). Lihat [00-concept/README.md](./00-concept/README.md) §Security untuk detail.

## Kategori dokumen

| Folder | Isi | Kapan dibaca |
|---|---|---|
| [00-concept/](./00-concept/) | Ringkasan sistem per role (PM, PD, UI/UX, FE/BE, Data Engineer, AI Engineer, Security, DevOps) | Onboarding pertama kali |
| [02-technical/](./02-technical/) | Detail teknis: alur sistem, config, tiap script, format label, referensi path | Kerja implementasi/training sehari-hari |

Dokumen kunci untuk mulai:

- [PANDUAN-MULAI-DARI-NOL.md](./PANDUAN-MULAI-DARI-NOL.md) — urutan kerja, keamanan data, dan penjelasan output.
- [00-concept/README.md](./00-concept/README.md) — ringkasan 1 menit + per role.
- [02-technical/ARSITEKTUR-DAN-SCRIPT.md](./02-technical/ARSITEKTUR-DAN-SCRIPT.md) — alur sistem lengkap, referensi tiap script.

## Hubungan dengan ocr-kia-reader

`scripts/predict_to_labels.py` di repo ini dipanggil langsung oleh `ocr-kia-reader/infra/nightly-pipeline/01-yolo-autolabel.sh` untuk auto-label produksi. Model hasil training di sini (`best.pt`) di-deploy manual ke sistem OCR di repo `ocr-kia-reader`. Tidak ada shared code/database — komunikasinya lewat file model yang di-copy.

## Riwayat

Dokumen ini menggantikan `DOKUMENTASI_TEKNIS.md` (sebelumnya di root repo, belum pernah di-commit) — isinya dipecah menjadi `00-concept/README.md` (ringkasan per role) dan `02-technical/ARSITEKTUR-DAN-SCRIPT.md` (detail teknis §1–12).
