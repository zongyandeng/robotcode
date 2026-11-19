# 重新繪製流程圖。所有文字集中在 T 字典，改字後重跑即可。
from PIL import Image, ImageDraw, ImageFont
import math
T = dict(
 s1="機器人電源啟動與系統初始化",
 s2="進入比賽操作流程",
 d1="自主階段或自動模式？",
 manual=["手動操作","移動、瞄準、投球與動作"],
 yolo=["YOLOv11n 偵測目標","Gemini 深度分析與車體導航"],
 d2="目標與深度資訊有效？",
 mlp=["MLP 預測彈道與落點","並用可用的 RNN 補償"],
 aim="水平對位、俯仰與轉速控制",
 d3="對位、俯仰與轉速到位？",
 fire=["執行發射並紀錄結果","更新參數與狀態並回到偵測"],
 note=["IT 模式發球需符合條件","停止動力輸出","清除異常後再回復化"],
 foot="註：以上為簡化示意流程圖",
 cap="圖：YOLOv11n 目標偵測與自動瞄準發射流程",
 yes="是", no="否")
S=2  # 超取樣
W,H=1544,1700
im=Image.new("RGB",(W*S,H*S),"white"); d=ImageDraw.Draw(im)
FONT="C:/Windows/Fonts/msjh.ttc"
def F(sz): return ImageFont.truetype(FONT,sz*S)
INK=(43,58,85); FILL=(236,240,248); BOX=(245,247,250)
def P(p): return (p[0]*S,p[1]*S)
def text(cx,cy,s,sz=30,fill=(20,25,35)):
    d.text((cx*S,cy*S),s,font=F(sz),fill=fill,anchor="mm")
def lines(cx,cy,ls,sz=30,gap=46):
    y=cy-gap*(len(ls)-1)/2
    for l in ls: text(cx,y,l,sz); y+=gap
def rbox(x0,y0,x1,y1):
    d.rounded_rectangle([x0*S,y0*S,x1*S,y1*S],radius=22*S,fill=BOX,outline=INK,width=3*S)
def dash(a,b,seg=14,gap=9,w=3):
    L=math.dist(a,b); n=int(L//(seg+gap))+1
    for i in range(n):
        t0=i*(seg+gap)/L; t1=min(1,(i*(seg+gap)+seg)/L)
        if t0>=1: break
        d.line([P((a[0]+(b[0]-a[0])*t0,a[1]+(b[1]-a[1])*t0)),P((a[0]+(b[0]-a[0])*t1,a[1]+(b[1]-a[1])*t1))],fill=INK,width=w*S)
def diamond(cx,cy,hw,hh,s):
    pts=[(cx,cy-hh),(cx+hw,cy),(cx,cy+hh),(cx-hw,cy)]
    d.polygon([P(p) for p in pts],fill=FILL)
    for i in range(4): dash(pts[i],pts[(i+1)%4])
    text(cx,cy,s,30)
def line(pts,arrow=True):
    d.line([P(p) for p in pts],fill=INK,width=3*S,joint="curve")
    if arrow:
        (x0,y0),(x1,y1)=pts[-2],pts[-1]; a=math.atan2(y1-y0,x1-x0); r=14
        tri=[(x1,y1),(x1-r*math.cos(a-.4),y1-r*math.sin(a-.4)),(x1-r*math.cos(a+.4),y1-r*math.sin(a+.4))]
        d.polygon([P(p) for p in tri],fill=INK)
CX=587
rbox(290,33,885,122);  text(CX,78,T["s1"])
rbox(290,170,885,265); text(CX,218,T["s2"])
diamond(CX,390,295,80,T["d1"])
rbox(1055,325,1487,457); lines(1271,391,T["manual"])
rbox(265,530,910,645); lines(CX,588,T["yolo"])
diamond(CX,760,275,72,T["d2"])
rbox(265,885,910,1002); lines(CX,944,T["mlp"])
rbox(265,1045,910,1150); text(CX,1097,T["aim"])
diamond(CX,1265,275,72,T["d3"])
rbox(265,1387,910,1502); lines(CX,1445,T["fire"])
d.rounded_rectangle([1055*S,662*S,1487*S,800*S],radius=22*S,fill=BOX,outline=INK,width=3*S)
lines(1271,731,T["note"],sz=26,gap=40)
# 連線
line([(CX,122),(CX,170)]); line([(CX,265),(CX,310)])
line([(CX+295,390),(1055,390)]); text(970,365,T["no"],26)
line([(CX,470),(CX,530)]); text(625,497,T["yes"],26)
line([(CX,645),(CX,688)])
line([(CX-275,760),(165,760),(165,585),(265,585)]); text(205,727,T["no"],26)
line([(CX,832),(CX,885)]); text(625,858,T["yes"],26)
line([(CX,1002),(CX,1045)]); line([(CX,1150),(CX,1193)])
line([(CX+275,1265),(978,1265),(978,1097),(910,1097)]); text(930,1235,T["no"],26)
line([(CX,1337),(CX,1387)]); text(625,1362,T["yes"],26)
line([(265,1445),(68,1445),(68,390),(CX-295,390)])
text(300,1552,T["foot"],24)
text(770,1635,T["cap"],26)
im=im.resize((W,H),Image.LANCZOS); im.save("flowchart_yolov11n_v2.png")
