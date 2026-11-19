#!/bin/bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
apt-get download tesseract-ocr-deu=1:4.1.0-2build1
pkg=tesseract-ocr-deu_1%3a4.1.0-2build1_all.deb
if [[ ! -f "$pkg" ]]; then pkgs=(./*.deb); pkg=${pkgs[0]}; fi
echo "9fefa09dfe8fdde390a17259b9df3e2fe1fd6294d1ccad9f25a06371f3fc4501  $pkg" | sha256sum -c -
dpkg-deb -x "$pkg" extracted
mkdir context
cp "$src"/{Dockerfile,pdf_page_ocr.py,docling-serve-1.12.patch} context/
cp extracted/usr/share/tesseract-ocr/*/tessdata/deu.traineddata context/
docker build --pull=false --network=none -t "${DOCLING_IMAGE_TAG:-kiara/docling-serve-cu126:1.12.0-page-ocr-2}" context
