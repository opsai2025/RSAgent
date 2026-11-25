from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import json

ROOT = Path(__file__).resolve().parent
def font(size, bold=False):
    names = (['C:/Windows/Fonts/segoeuib.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
             if bold else ['C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)

def rounded_image(path, size):
    img = Image.open(path).convert('RGB')
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, '#101a32')
    canvas.paste(img, ((size[0]-img.width)//2,(size[1]-img.height)//2))
    return canvas

def make_building():
    folder = ROOT/'building'
    data = json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
    canvas = Image.new('RGB',(1600,1020),'#111b38')
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((40,38,1560,980), radius=32, fill='#18294c', outline='#3a568d', width=2)
    d.text((76,69),'REAL IMAGE  /  OFFLINE EXAMPLE',font=font(21,True),fill='#72e7cc')
    d.text((76,109),'Find the largest building and create a mask',font=font(40,True),fill='white')
    d.text((76,168),f"{Path(data['source']).name}  •  UniBuild prediction  •  largest connected component by pixel area",font=font(22),fill='#c3d4ef')
    panel_size=(700,540)
    for x, label, filename, color in [(76,'01  SOURCE IMAGE','input.jpg','#85bcff'),(824,'02  SELECTED MASK OVERLAY','overlay.jpg','#ffb27a')]:
        d.text((x,225),label,font=font(21,True),fill=color)
        canvas.paste(rounded_image(folder/filename,panel_size),(x,263))
    d.rounded_rectangle((76,834,1524,936),radius=16,fill='#213960')
    d.text((100,850),f"{data['predicted_components']} predicted components  →  selected {data['selected_component']['area_px']:,} pixels",font=font(25,True),fill='white')
    d.text((100,890),'Offline: VLM stubbed; component ranked by pixels. Adjacent roofs may merge.',font=font(19),fill='#d6e4f7')
    canvas.save(folder/'example-card.png',optimize=True)
    frames=[]
    for label, filename in [('SOURCE IMAGE','input.jpg'),('ACTUAL LARGEST-COMPONENT OVERLAY','overlay.jpg')]:
        frame=Image.new('RGB',(920,790),'#111b38'); dr=ImageDraw.Draw(frame)
        dr.rounded_rectangle((20,20,900,770),radius=22,fill='#18294c')
        dr.text((45,42),label,font=font(23,True),fill='#7ae7cc')
        frame.paste(rounded_image(folder/filename,(830,632)),(45,83))
        dr.text((45,732),f"Offline UniBuild  •  {Path(data['source']).name}  •  VLM stubbed",font=font(18),fill='#d6e4f7')
        frames.append(frame.convert('P',palette=Image.Palette.ADAPTIVE,colors=256))
    frames[0].save(folder/'example.gif',save_all=True,append_images=frames[1:],duration=[1800,2300],loop=0,optimize=True)
    print('Created building card and GIF from actual artifacts')

def make_landcover():
    folder = ROOT/'landcover'
    mapping = json.loads((folder/'classification_map.json').read_text(encoding='utf-8-sig'))
    canvas = Image.new('RGB',(1600,1100),'#111b38')
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((40,38,1560,1060),radius=32,fill='#18294c',outline='#3a568d',width=2)
    d.text((76,69),'REAL IMAGE  /  OFFLINE EXAMPLE',font=font(21,True),fill='#72e7cc')
    d.text((76,109),'Create a land-use / land-cover classification',font=font(40,True),fill='white')
    d.text((76,168),'P0127.png  •  SegEarth-OV-3 + UniBuild  •  local rule-validated plan',font=font(22),fill='#c3d4ef')
    for x,label,name,color in [(76,'01  SOURCE IMAGE','image.png','#85bcff'),
                               (824,'02  CLASSIFICATION MASK','classification_map.png','#ffb27a')]:
        d.text((x,225),label,font=font(21,True),fill=color)
        canvas.paste(rounded_image(folder/name,(700,540)),(x,263))
    d.rounded_rectangle((76,834,1524,1017),radius=16,fill='#213960')
    for i,entry in enumerate(mapping['class_table']):
        x=100+(i%4)*350; y=856+(i//4)*43
        color=tuple(entry['color'])
        d.rounded_rectangle((x,y+3,x+18,y+21),radius=3,fill=color,outline='#b9c8dc')
        label=f"{entry['name']} {entry['coverage_percent']:.1f}%"
        d.text((x+28,y-2),label,font=font(18,True),fill='white')
    d.text((100,970),'Pixel coverage from the model output; no live router or VLM call.',font=font(19),fill='#d6e4f7')
    canvas.save(folder/'example-card.png',optimize=True)
    print('Created land-cover card from actual artifacts')

if __name__=='__main__':
    make_building()
    if (ROOT/'landcover'/'classification_map.json').is_file():
        make_landcover()
