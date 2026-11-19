FROM quay.io/docling-project/docling-serve-cu126@sha256:dc3beb29b286cb6e4b8839f6d5ca1d792a5052c4ec93d4d41d4a2da073c6e3a9
USER 0
COPY docling-serve-1.12.patch /tmp/docling-serve.patch
RUN patch -d / -p1 --fuzz=0 < /tmp/docling-serve.patch && rm /tmp/docling-serve.patch
COPY pdf_page_ocr.py /opt/app-root/src/docling_serve/pdf_page_ocr.py
COPY deu.traineddata /usr/share/tesseract/tessdata/deu.traineddata
USER 1001
