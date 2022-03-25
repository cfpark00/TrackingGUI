from PyQt5.QtWidgets import *
import pyqtgraph as pg

class FigWidget(pg.PlotWidget,QGraphicsItem):
    def __init__(self,id,lab,im,parent):
        super().__init__()
        self.parent=parent
        self.id=id
        self.setMenuEnabled(False)
        self.invertY(True)
        self.setAspectLocked(True)
        self.setContentsMargins(0,0,0,0)

        self.scene().sigMouseClicked.connect(self.mouseclick)

        self.img=pg.ImageItem(im)
        self.addItem(self.img)
        self.ptx=[]
        self.pty=[]
        self.scat=pg.ScatterPlotItem()
        self.addItem(self.scat)
        self.hideAxis('bottom')
        self.hideAxis('left')

        self.label=pg.TextItem(lab)
        self.addItem(self.label)

        self.inn=True

    def mouseclick(self,event):
        pos=self.plotItem.vb.mapSceneToView(event.scenePos())
        print("sending",self.id,pos.x()-0.5,pos.y()-0.5)
        self.parent.receive_pt(self.id,pos.x()-0.5,pos.y()-0.5)

    def enterEvent(self, QEvent):
        self.setFocus()
        self.inn=True

    def leaveEvent(self, QEvent):
        self.inn=False

    def addpt(self,pt):
        self.ptx.append(pt[0]+0.5)
        self.pty.append(pt[1]+0.5)
        self.scat.setData(x=self.ptx,y=self.pty,brush="r" if self.id=="from" else "g")


class GetPtsWidget(QDialog):
    def __init__(self,name,imfrom,imto,ptsfrom,ptsto):
        super().__init__()
        self.setWindowTitle(name)
        self.ptsfrom=ptsfrom
        self.ptsto=ptsto
        self.ptstemp={}
        self.ptstemp["from"]=[]
        self.ptstemp["to"]=[]
        lay=QGridLayout()
        savebut=QPushButton("Validate")
        savebut.clicked.connect(self.register)
        savebut.setStyleSheet("background-color: green;")
        cancelbut=QPushButton("Abort")
        cancelbut.clicked.connect(self.close)
        cancelbut.setStyleSheet("background-color: red;")
        self.Ims={}
        self.Ims["from"]=FigWidget("from","Red (From)",imfrom,self)
        lay.addWidget(self.Ims["from"],0,0)
        self.Ims["to"]=FigWidget("to","Green (To)",imto,self)
        lay.addWidget(self.Ims["to"],0,1)
        lay.addWidget(savebut,1,0)
        lay.addWidget(cancelbut,1,1)
        self.setLayout(lay)
        self.selectionfrom=True
        self.show()

    def receive_pt(self,id,x,y):
        #print("rec",id,x,y)
        if (id=="from")==self.selectionfrom:#if they are same
            #print("in")
            self.selectionfrom=not self.selectionfrom
            self.ptstemp[id].append([x,y])
            self.Ims[id].addpt([x,y])

    def register(self):
        if len(self.ptstemp["from"])!=len(self.ptstemp["to"]):
            print("Points not matching!")
            return
        #print("Sending from",self.ptstemp["from"])
        #print("Sending to",self.ptstemp["to"])
        self.ptsfrom.extend(self.ptstemp["from"])
        self.ptsto.extend(self.ptstemp["to"])
        self.close()

class Fig(pg.PlotWidget,QGraphicsItem):
    def __init__(self,lab,im):
        super().__init__()
        self.setMenuEnabled(False)
        self.invertY(True)
        self.setAspectLocked(True)
        self.setContentsMargins(0,0,0,0)
        self.img=pg.ImageItem(im)
        self.addItem(self.img)
        self.label=pg.TextItem(lab)
        self.addItem(self.label)

class ConfirmAlWidget(QDialog):
    def __init__(self,name,ims,result):
        super().__init__()
        self.setWindowTitle(name)
        self.result=result
        lay=QGridLayout()
        savebut=QPushButton("Validate")
        savebut.clicked.connect(self.register)
        savebut.setStyleSheet("background-color: green;")
        cancelbut=QPushButton("Abort")
        cancelbut.clicked.connect(self.close)
        cancelbut.setStyleSheet("background-color: red;")
        row=3
        self.ims={}
        c=0
        for key,im in ims.items():
            self.ims[key]=Fig(key,im)
            lay.addWidget(self.ims[key],c//row,c%row)
            c+=1
        newrow=(c-1)//row+1
        lay.addWidget(savebut,newrow,0)
        lay.addWidget(cancelbut,newrow,1)

        self.setLayout(lay)
        self.show()

    def register(self):
        self.result[0]=True
        self.close()
