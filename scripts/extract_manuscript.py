"""Read-only DOCX extraction preserving native OMML structure and source hash."""
import argparse, hashlib, json, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
def render(e):
    tag=e.tag.split('}')[-1]
    def child(name):
        x=e.find(M+name)
        return render(x) if x is not None else ''
    if tag in ('t',): return e.text or ''
    if tag.endswith('Pr'): return ''
    if tag=='f': return '('+child('num')+')/('+child('den')+')'
    if tag=='sSub': return child('e')+'_{'+child('sub')+'}'
    if tag=='sSup': return child('e')+'^{'+child('sup')+'}'
    if tag=='sSubSup': return child('e')+'_{'+child('sub')+'}^{'+child('sup')+'}'
    if tag=='rad': return 'root['+child('deg')+']('+child('e')+')'
    if tag=='nary':
        c=e.find(M+'naryPr/'+M+'chr')
        op=c.get(M+'val') if c is not None else 'integral'
        return op+'_{'+child('sub')+'}^{'+child('sup')+'}('+child('e')+')'
    if tag=='d': return '('+''.join(render(x) for x in e if x.tag==M+'e')+')'
    if tag=='m': return '['+'; '.join(render(x) for x in e if x.tag==M+'mr')+']'
    if tag=='mr': return ', '.join(render(x) for x in e)
    if tag in ('bar','acc'):
        p=e.find(M+tag+'Pr/'+M+'chr')
        symbol=p.get(M+'val') if p is not None else tag
        return symbol+'('+child('e')+')'
    return ''.join(render(x) for x in e)
def extract(source, out):
    data=Path(source).read_bytes()
    with zipfile.ZipFile(source) as z: xml=z.read('word/document.xml')
    root=ET.fromstring(xml)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    lines=[]
    for i,p in enumerate(root.iter(W+'p')):
        s=render(p)
        if s.strip(): lines.append(f'[{i:04d}] {s}')
    (out/'manuscript_extracted.txt').write_text('\n'.join(lines),encoding='utf-8')
    formulas=[ET.tostring(e,encoding='unicode') for e in root.iter(M+'oMath')]
    (out/'equations_omml.json').write_text(json.dumps(formulas,ensure_ascii=False,indent=2),encoding='utf-8')
    metadata={'source':str(source),'sha256':hashlib.sha256(data).hexdigest(),'native_equation_count':len(formulas),'paragraph_count':len(lines)}
    (out/'source_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--out',default='docs/source'); a=p.parse_args(); extract(a.source,a.out)
