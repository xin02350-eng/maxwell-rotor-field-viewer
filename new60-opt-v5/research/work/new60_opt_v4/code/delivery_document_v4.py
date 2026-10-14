"""Small standard-library editable Chinese Word/Markdown renderer."""
import re
import zipfile
from xml.sax.saxutils import escape

def plain(text):
    text=re.sub(r'\[([^]]*)\]\(([^)]+)\)',r'\1（\2）',text)
    text=re.sub(r'(?m)^#{1,6}\s*','',text)
    return text.replace('**','').replace('`','')

def docx(path,markdown):
    def paragraph(text,style=None):
        prop=f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        return '<w:p>'+prop+'<w:r><w:t xml:space="preserve">'+escape(plain(text))+'</w:t></w:r></w:p>'
    body=[];pending=[]
    def flush():
        if not pending:return
        rows=[]
        for line in pending:
            cells=re.split(r'(?<!\\)\|',line.strip().strip('|'))
            if all(re.fullmatch(r'\s*:?-+:?\s*',c) for c in cells):continue
            rows.append('<w:tr>'+''.join('<w:tc>'+paragraph(c.strip().replace('\\|','|'))+'</w:tc>' for c in cells)+'</w:tr>')
        body.append('<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>'+''.join(rows)+'</w:tbl>');pending.clear()
    for line in markdown.splitlines():
        if line.startswith('|'):pending.append(line);continue
        flush()
        if not line.strip():continue
        h=re.match(r'^(#{1,3})\s+(.*)',line)
        body.append(paragraph(h[2],'Title' if len(h[1])==1 else 'Heading1') if h else paragraph(line))
    flush();ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    document='<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="'+ns+'"><w:body>'+''.join(body)+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="850"/></w:sectPr></w:body></w:document>'
    styles='<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="'+ns+'"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="宋体"/><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>'
    for name,size in [('Title',34),('Heading1',28)]:styles+=f'<w:style w:type="paragraph" w:styleId="{name}"><w:name w:val="{name}"/><w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr></w:style>'
    styles+='</w:styles>'
    ct='<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    rel='<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    dr='<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in [('[Content_Types].xml',ct),('_rels/.rels',rel),('word/document.xml',document),('word/styles.xml',styles),('word/_rels/document.xml.rels',dr)]:z.writestr(name,data.encode())
