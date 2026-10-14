"""Use the skill renderer's PNG workflow with a native Word-exported PDF.

The Windows runtime has no bundled LibreOffice. No user LibreOffice is used.
"""
from pathlib import Path
import importlib.util
import tempfile
from pdf2image import convert_from_path
ROOT = Path(__file__).resolve().parents[1]
SKILL = Path('C:/Users/LENOVO/.codex/plugins/cache/openai-primary-runtime/documents/26.1011.11113/skills/documents')
qa = ROOT/'outputs/documentation_word/qa'
temp_root = qa/'temp'
temp_root.mkdir(exist_ok=True)
assert temp_root.resolve().is_relative_to((ROOT/'outputs').resolve())
tempfile.tempdir = str(temp_root)
spec=importlib.util.spec_from_file_location('document_renderer',SKILL/'render_docx.py')
renderer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
pdf=qa/'preview.pdf'
assert pdf.is_file()
renderer.convert_to_pdf=lambda *args,**kwargs:(str(pdf),'Native Microsoft Word PDF export')
poppler='C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
renderer.convert_from_path=lambda *args,**kwargs:convert_from_path(*args,poppler_path=poppler,**kwargs)
pages=renderer.rasterize(str(ROOT/'outputs/documentation_word/RBMF_CS_Project_Documentation_20261014.docx'),
                         str(qa),130,False,False)
print(f'Rendered {len(pages)} page PNGs using Word PDF and bundled Poppler.')
