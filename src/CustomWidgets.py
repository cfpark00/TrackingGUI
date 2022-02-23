from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import QRect, QPoint, Qt
import pyqtgraph as pg
import numpy as np
import time
from src.methods import analysis_methods
from src.methods import tracking_methods
import threading

class PointBarWidget(QScrollArea):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.widget=QWidget()
        self.available_keys=self.gui.settings["keys"].split(",")

        self.grid=QGridLayout()
        self.widget.setLayout(self.grid)
        self.N_points=self.gui.data_info["N_points"]
        self.pointbuttons={}
        for i_point in range(1,self.N_points+1):
            indicator_button=QPushButton(str(i_point))
            indicator_button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px; min-width: 10px; min-height: 20px")
            indicator_button.clicked.connect(self.make_click_select_function(i_point))
            key_button=QPushButton()
            key_button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px; min-width: 10px; min-height: 20px")
            key_button.clicked.connect(self.make_assign_key_function(i_point))
            self.grid.addWidget(indicator_button,0,i_point-1)
            self.grid.addWidget(key_button,1,i_point-1)
            self.pointbuttons[i_point]=[indicator_button,key_button]

        self.setWidget(self.widget)
        self.setMinimumHeight(self.sizeHint().height()+self.horizontalScrollBar().sizeHint().height())
        self.verticalScrollBar().setEnabled(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def make_assign_key_function(self,i_point):
        def assign_key_function():
            key,ok=QInputDialog.getText(self,"Get Point Key","Insert Key for Point "+str(i_point))
            if key not in self.available_keys:
                ok=False
            if ok:
                self.gui.respond("update_assignments",[key,i_point])
        return assign_key_function

    def make_click_select_function(self,i_point):
        def click_select_function():
            self.gui.respond("highlight",i_point)
        return click_select_function

    def update_assigned(self):
        for i_point in range(1,self.N_points+1):
            self.pointbuttons[i_point][1].setText("")
        for key,i_point in self.gui.assigned_points.items():
            if i_point is not None:
                self.pointbuttons[i_point][1].setText(key)

    def recolor(self,presence):
        for i_point in range(1,self.N_points+1):
            present=presence[i_point]
            if present==0:
                self.pointbuttons[i_point][0].setStyleSheet("background-color : rgb(236,98,64); border-radius: 4px; min-width: 10px; min-height: 20px")
            elif present==1:
                self.pointbuttons[i_point][0].setStyleSheet("background-color : rgb(42,99,246); border-radius: 4px; min-width: 10px; min-height: 20px")
            elif present==2:
                self.pointbuttons[i_point][0].setStyleSheet("background-color : rgb(243,175,61); border-radius: 4px; min-width: 10px; min-height: 20px")
            elif present==3:
                self.pointbuttons[i_point][0].setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-width: 10px; min-height: 20px")
                
class Annotated3DFig(pg.PlotWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.N_assignable=len(self.gui.settings["keys"].split(","))
        self.assigned_colors=np.array([[int(val) for val in col.split(",")]  for col in self.gui.settings["keys_colors"].split(";")])
        self.N_points=self.gui.data_info["N_points"]
        self.channel_colors=[[int(val) for val in chan.split(",")]  for chan in self.gui.settings["channel_colors"].split(";")]
        self.channel_colors=np.array(self.channel_colors)
        self.n_channels=self.gui.data_info["C"]
        if self.n_channels>self.channel_colors.shape[0]:
            print("Not enough channel colors given, setting as dark")
            self.channel_colors=np.concatenate([self.channel_colors,np.full((self.n_channels-self.channel_colors.shape[0],3),0)],axis=0)
        elif self.n_channels<self.channel_colors.shape[0]:
            self.channel_colors=self.channel_colors[:self.n_channels]

        self.image=pg.ImageItem()
        self.addItem(self.image)

        self.label=pg.TextItem()
        self.addItem(self.label)
        
        self.scatter=pg.ScatterPlotItem()
        self.scatter_highlight=pg.ScatterPlotItem()
        pw=float(self.gui.settings["pen_width"])
        self.pens_assigned=[pg.mkPen(width=pw, color=color) for color in self.assigned_colors]
        self.pen_gt=pg.mkPen(width=pw, color=[int(val) for val in self.gui.settings["gt_color"].split(",")]) 
        self.pen_helper=pg.mkPen(width=pw, color=[int(val) for val in self.gui.settings["helper_color"].split(",")])
        self.pen_highlight=pg.mkPen(width=pw*3, color=(255,255,255))
        mp=float(self.gui.settings["min_pointsize"])
        Mp=float(self.gui.settings["max_pointsize"])
        zss=float(self.gui.settings["z_size_scale"])
        c1=(Mp-mp)/zss
        self.size_func=lambda dz: np.clip(Mp-c1*dz,mp,Mp)
        self.addItem(self.scatter)
        
        self.setAspectLocked()
        self.setMenuEnabled(False)
        self.invertY(False)
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.transposed=False
        
        self.scene().sigMouseMoved.connect(self.mousetracker)
        self.scene().sigMouseClicked.connect(self.mouseclick)
        self.inn=False
        self.mouse_current=None
        
        self.levels=np.repeat(np.array([[0,255]]),self.n_channels,axis=0)[:,:,None,None]
        self.gammas=np.ones(self.n_channels,dtype=np.float32)[:,None,None]

    def update_data(self,data):
        z=data["z"]
                
        im=data["image"]
        im=np.divide((im-self.levels[:,0]),(self.levels[:,1]-self.levels[:,0]),
            where=self.levels[:,1]>self.levels[:,0],out=np.zeros_like(im,dtype=np.float32))
        im=np.clip(im,0,1)
        im=im**self.gammas
        #print(np.min(im),np.max(im),self.gammas)
        if self.transposed:
            im=im.transpose(2,1,0)@self.channel_colors
        else:
            im=im.transpose(1,2,0)@self.channel_colors
        self.image.setImage(im[:,:,:],autoLevels=False,levels=[0,255])
        
        points=data["points"]
        valid=~np.isnan(points[:,0])
        if np.sum(valid)==0:
            self.scatter.clear()

        coords=points[valid,:3]

        pt_type=points[:,3]
        pens=np.empty(self.N_points+1,dtype=object)
        pens[pt_type==-1]=self.pen_gt
        pens[pt_type==-2]=self.pen_helper
        for i,pen in enumerate(self.pens_assigned):
            pens[pt_type==i]=pen
        pens[pt_type==-3]=self.pen_highlight
        pens=pens[valid]

        if self.transposed:
            self.scatter.setData(x=coords[:,1]+0.5,y=coords[:,0]+0.5,pen=pens,brush=None,
                size=self.size_func(np.abs(coords[:,2]-z)),pxMode=True)
        else:
            self.scatter.setData(x=coords[:,0]+0.5,y=coords[:,1]+0.5,pen=pens,brush=None,
                size=self.size_func(np.abs(coords[:,2]-z)),pxMode=True)
        self.label.setText(data["label"])
        self.plotItem.vb.disableAutoRange()

    
    def update_params(self,arg):
        code,val=arg
        key,channel=code.split("_")
        channel=int(channel)
        if key=="min":
            self.levels[channel-1,0]=val
        elif key=="max":
            self.levels[channel-1,1]=val
        elif key=="gamma":
            self.gammas[channel-1]=val/100
        elif key=="r":
            self.channel_colors[channel-1,0]=val
        elif key=="g":
            self.channel_colors[channel-1,1]=val
        elif key=="b":
            self.channel_colors[channel-1,2]=val
            
    def enterEvent(self,event):
        self.setFocus()
        self.inn=True

    def leaveEvent(self,event):
        self.inn=False
        
    def mousetracker(self,event):
        if self.inn:
            self.mouse_current=event
    
    def mouseclick(self,event):
        if self.inn:
            pos=self.plotItem.vb.mapSceneToView(event.scenePos())
            if self.transposed:
                self.gui.respond("fig_click",[event.button(),pos.y()-0.5,pos.x()-0.5])
            else:
                self.gui.respond("fig_click",[event.button(),pos.x()-0.5,pos.y()-0.5])
    
    def wheelEvent(self,event):
        self.gui.respond("fig_scroll",event.angleDelta().y())
        
    def keyPressEvent(self, event):
        key=event.key()
        if key==Qt.Key_Space:
            self.plotItem.vb.enableAutoRange()
        else:
            if self.inn and (self.mouse_current is not None):
                pos=self.plotItem.vb.mapSceneToView(self.mouse_current)
                if self.transposed:
                    self.gui.respond("fig_keypress",[event.text(),pos.y()-0.5,pos.x()-0.5])
                else:
                    self.gui.respond("fig_keypress",[event.text(),pos.x()-0.5,pos.y()-0.5])
                    
    def set_transpose(self,val):
        self.transposed=val
        self.enableAutoRange()         
        
class AnnotateTab(QWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        T=self.gui.data_info["T"]
        self.grid=QGridLayout()
        n_cols=3
        
        row=0
        self.label=QLabel("Selected: None")
        self.grid.addWidget(self.label,row,0,1,n_cols)
        row+=1
        
        self.helper_select=QComboBox()
        self.helper_names=[""]+self.gui.dataset.get_helper_names()
        for name in self.helper_names:
            self.helper_select.addItem(name)
        self.helper_select.setCurrentIndex(0)
        self.helper_select.currentIndexChanged.connect(lambda x:self.gui.respond("load_helper",self.helper_names[x]))
        self.grid.addWidget(self.helper_select,row,0,1,n_cols)
        row+=1
        
        self.grid.addWidget(QLabel("Action"),row,0,1,1)
        self.grid.addWidget(QLabel("Tmin"),row,1,1,1)
        self.grid.addWidget(QLabel("Tmax"),row,2,1,1)
        row+=1
        
        self.approve_button=QPushButton("Approve Helper")
        self.approve_button.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-height: 20px")
        self.approve_button.setEnabled(False)
        self.approve_button.clicked.connect(self.make_annotate_signal_func("approve"))
        self.grid.addWidget(self.approve_button,row,0,1,1)
        self.amin=QLineEdit("1")
        self.amin.setValidator(QtGui.QIntValidator(1,T))
        self.grid.addWidget(self.amin,row,1,1,1)
        self.amax=QLineEdit(str(T))
        self.amax.setValidator(QtGui.QIntValidator(1,T))
        self.grid.addWidget(self.amax,row,2,1,1)
        row+=1
        
        self.extend_button=QPushButton("Linear Interpolate GT")
        self.extend_button.setStyleSheet("background-color : rgb(243,175,61); border-radius: 4px; min-height: 20px")
        self.extend_button.setEnabled(False)
        self.extend_button.clicked.connect(self.make_annotate_signal_func("lin_intp"))
        self.grid.addWidget(self.extend_button,row,0,1,1)
        self.emin=QLineEdit("1")
        self.emin.setValidator(QtGui.QIntValidator(1,T))
        self.grid.addWidget(self.emin,row,1,1,1)
        self.emax=QLineEdit(str(T))
        self.emax.setValidator(QtGui.QIntValidator(1,T))
        self.grid.addWidget(self.emax,row,2,1,1)
        row+=1
        
        self.delete_button=QPushButton("Delete GT")
        self.delete_button.setStyleSheet("background-color : rgb(236,98,64); border-radius: 4px; min-height: 20px")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.make_annotate_signal_func("delete"))
        self.grid.addWidget(self.delete_button,row,0,1,1)
        self.dmin=QLineEdit("1")
        self.dmin.setValidator(QtGui.QIntValidator(1,T))
        self.grid.addWidget(self.dmin,row,1,1,1)
        self.dmax=QLineEdit(str(T))
        self.dmax.setValidator(QtGui.QIntValidator(1,T))
        self.grid.addWidget(self.dmax,row,2,1,1)
        row+=1
        
        follow_checkbox=QCheckBox("Follow Highlighted")
        follow_checkbox.stateChanged.connect(self.follow_clicked)
        self.grid.addWidget(follow_checkbox,row,0,1,n_cols)
        
        self.setLayout(self.grid)
    
    def follow_clicked(self,state):
        if state == QtCore.Qt.Checked:
            self.gui.respond("follow",True)
        else:
            self.gui.respond("follow",False)
            
    def highlight(self,i_point_highlight):
        if i_point_highlight!=0:
            self.label.setText("Selected: "+str(i_point_highlight))
            self.approve_button.setEnabled(True)
            self.extend_button.setEnabled(True)
            self.delete_button.setEnabled(True)
        else:
            self.label.setText("Selected: None")
            self.approve_button.setEnabled(False)
            self.extend_button.setEnabled(False)
            self.delete_button.setEnabled(False)
        
    def make_annotate_signal_func(self,code):
        def annotate_signal_func():
            if code=="approve":
                mM=[self.amin.text(),self.amax.text()]
            elif code=="lin_intp":
                mM=[self.emin.text(),self.emax.text()]
            elif code=="delete":
                mM=[self.dmin.text(),self.dmax.text()]
            try:
                mM=[int(txt) for txt in mM]
            except:
                pass
                #print("Range invalid")
            self.gui.respond(code,mM)
        return annotate_signal_func
        
    def get_current_helper_name(self):
        return self.helper_names[self.helper_select.currentIndex()]
    
    def renew_helper_list(self):
        self.helper_select.currentIndexChanged.disconnect()
        self.helper_select.clear()
        self.helper_names=[""]+self.gui.dataset.get_helper_names()
        for name in self.helper_names:
            self.helper_select.addItem(name)
        self.helper_select.setCurrentIndex(0)
        self.helper_select.currentIndexChanged.connect(lambda x:self.gui.respond("load_helper",self.helper_names[x]))
        
class ViewTab(QWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.grid=QGridLayout()
        n_channels=self.gui.data_info["C"]
        channel_colors=[[int(val) for val in chan.split(",")]  for chan in self.gui.settings["channel_colors"].split(";")]
        channel_colors=np.array(channel_colors)
        if n_channels>channel_colors.shape[0]:
            channel_colors=np.concatenate([channel_colors,np.full((n_channels-channel_colors.shape[0],3),0)],axis=0)
        max_100gamma=int(float(self.gui.settings["max_gamma"])*100)
        
        row=0
        self.transpose=QCheckBox("Transpose")
        self.transpose.toggled.connect(lambda x:self.gui.respond("transpose",x))
        self.grid.addWidget(self.transpose,row,0,1,3*n_channels)
        row+=1
        
        for i_channel in range(1,n_channels+1):
            c=channel_colors[i_channel-1]
            if True:
                subrow=row
                label=QLabel("Channel "+str(i_channel))
                self.grid.addWidget(label,subrow,3*(i_channel-1),1,3)
                subrow+=1
                
                label=QLabel("Min")
                self.grid.addWidget(label,subrow,3*(i_channel-1),1,3)
                subrow+=1
                min_slider=QSlider(Qt.Horizontal)
                min_slider.setMinimum(0)
                min_slider.setMaximum(255)
                min_slider.setValue(0)
                min_slider.valueChanged.connect(self.make_slider_change_func("min_"+str(i_channel)))
                self.grid.addWidget(min_slider,subrow,3*(i_channel-1),1,3)
                subrow+=1
                
                label=QLabel("Max")
                self.grid.addWidget(label,subrow,3*(i_channel-1),1,3)
                subrow+=1
                max_slider=QSlider(Qt.Horizontal)
                max_slider.setMinimum(0)
                max_slider.setMaximum(255)
                max_slider.setValue(255)
                max_slider.valueChanged.connect(self.make_slider_change_func("max_"+str(i_channel)))
                self.grid.addWidget(max_slider,subrow,3*(i_channel-1),1,3)
                subrow+=1
                
                label=QLabel("Gamma")
                self.grid.addWidget(label,subrow,3*(i_channel-1),1,3)
                subrow+=1
                gamma_slider=QSlider(Qt.Horizontal)
                gamma_slider.setMinimum(0)
                gamma_slider.setMaximum(max_100gamma)
                gamma_slider.setValue(100)
                gamma_slider.valueChanged.connect(self.make_slider_change_func("gamma_"+str(i_channel)))
                self.grid.addWidget(gamma_slider,subrow,3*(i_channel-1),1,3)
                subrow+=1
                
                label=QLabel("R")
                self.grid.addWidget(label,subrow,3*(i_channel-1),1,1)
                label=QLabel("G")
                self.grid.addWidget(label,subrow,3*(i_channel-1)+1,1,1)
                label=QLabel("B")
                self.grid.addWidget(label,subrow,3*(i_channel-1)+2,1,1)
                subrow+=1
                
                r_slider=QSlider(Qt.Vertical)
                r_slider.setMinimum(0)
                r_slider.setMaximum(255)
                r_slider.setValue(channel_colors[i_channel-1,0])
                r_slider.valueChanged.connect(self.make_slider_change_func("r_"+str(i_channel)))
                self.grid.addWidget(r_slider,subrow,3*(i_channel-1),1,1)

                g_slider=QSlider(Qt.Vertical)
                g_slider.setMinimum(0)
                g_slider.setMaximum(255)
                g_slider.setValue(channel_colors[i_channel-1,1])
                g_slider.valueChanged.connect(self.make_slider_change_func("g_"+str(i_channel)))
                self.grid.addWidget(g_slider,subrow,3*(i_channel-1)+1,1,1)

                b_slider=QSlider(Qt.Vertical)
                b_slider.setMinimum(0)
                b_slider.setMaximum(255)
                b_slider.setValue(channel_colors[i_channel-1,2])
                b_slider.valueChanged.connect(self.make_slider_change_func("b_"+str(i_channel)))
                self.grid.addWidget(b_slider,subrow,3*(i_channel-1)+2,1,1)
                subrow+=1
        self.setLayout(self.grid)
    
    def make_slider_change_func(self,code):
        def slider_change_func(val):
            self.gui.respond("view_change",[code,val])
        return slider_change_func

class TrackTab(QWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        
        self.methods=tracking_methods.methods
        
        row=0
        self.grid=QGridLayout()
        self.combobox=QComboBox()
        self.combobox.addItem("")
        for key in self.methods.keys():
            self.combobox.addItem(key)
        self.combobox.setCurrentIndex(0)
        self.combobox.currentIndexChanged.connect(lambda x: self.run_button.setEnabled(False) if x==0 else self.run_button.setEnabled(True))
        self.grid.addWidget(self.combobox,row,0)
        row+=1
        
        self.param_edit=QLineEdit()
        self.grid.addWidget(self.param_edit,row,0)
        row+=1
        
        self.run_button=QPushButton("Run")
        self.run_button.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-height: 20px")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.make_run_function())
        self.grid.addWidget(self.run_button,row,0)
        row+=1
        
        self.setLayout(self.grid)
    
    def make_run_function(self):
        def run_function():
            name=str(self.combobox.currentText())
            params=self.param_edit.text()
            self.run(name,params)
        return run_function
    
    def run(self,method_name,params):
        msgbox=QMessageBox()
        msgbox.setText("Confirm Run")
        msgbox.setInformativeText("Run "+method_name+" with "+params+"?")
        msgbox.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        res=msgbox.exec()
        if res==QMessageBox.No:
            return
        self.gui.respond("save")
        self.gui.respond("timer_stop")
        self.gui.dataset.close()
        print("Running",method_name,"with",params)
        try:
            method=self.methods[method_name](params)
            result=method(self.gui.dataset.file_path)
            print("Run Success")
        except Exception as ex:
            print(ex)
            result=ex
            print("Run Failed")
        self.gui.dataset.open()
        self.gui.respond("renew_helpers")
        self.gui.respond("timer_start")
        return result

class AnalysisTab(QWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        
        self.methods=analysis_methods.methods
        
        row=0
        self.grid=QGridLayout()
        self.combobox=QComboBox()
        self.combobox.addItem("")
        for key in self.methods.keys():
            self.combobox.addItem(key)
        self.combobox.setCurrentIndex(0)
        self.combobox.currentIndexChanged.connect(lambda x: self.run_button.setEnabled(False) if x==0 else self.run_button.setEnabled(True))
        self.grid.addWidget(self.combobox,row,0)
        row+=1
        
        self.param_edit=QLineEdit()
        self.grid.addWidget(self.param_edit,row,0)
        row+=1
        
        self.run_button=QPushButton("Run")
        self.run_button.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-height: 20px")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.make_run_function())
        self.grid.addWidget(self.run_button,row,0)
        row+=1
        
        self.signal_select=QComboBox()
        self.signal_names=[""]+self.gui.dataset.get_signal_names()
        for name in self.signal_names:
            self.signal_select.addItem(name)
        self.signal_select.setCurrentIndex(0)
        self.signal_select.currentIndexChanged.connect(lambda x:self.gui.respond("load_signal",self.signal_names[x]))
        self.grid.addWidget(self.signal_select,row,0)
        row+=1
        
        self.setLayout(self.grid)
    
    def make_run_function(self):
        def run_function():
            name=str(self.combobox.currentText())
            params=self.param_edit.text()
            self.run(name,params)
        return run_function
        
    def run(self,method_name,params):
        msgbox=QMessageBox()
        msgbox.setText("Confirm Run")
        msgbox.setInformativeText("Run "+method_name+" with "+params+"?")
        msgbox.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        res=msgbox.exec()
        if res==QMessageBox.No:
            return
        self.gui.respond("save")
        self.gui.respond("timer_stop")
        self.gui.dataset.close()
        print("Running",method_name,"with",params)
        
        try:
            method=self.methods[method_name](params)
            result=method(self.gui.dataset.file_path)
            print("Run Success")
        except Exception as ex:
            print(ex)
            result=ex
            print("Run Failed")
        self.gui.dataset.open()
        self.gui.respond("renew_signals")
        self.gui.respond("timer_start")
        return result
        
    def renew_signal_list(self):
        self.signal_select.currentIndexChanged.disconnect()
        self.signal_select.clear()
        self.signal_names=[""]+self.gui.dataset.get_signal_names()
        for name in self.signal_names:
            self.signal_select.addItem(name)
        self.signal_select.setCurrentIndex(0)
        self.signal_select.currentIndexChanged.connect(lambda x:self.gui.respond("load_signal",self.signal_names[x]))
        
class DashboardTab(QWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.T=int(self.gui.data_info["T"])
        self.scrollarea=QScrollArea()
        self.scrollwidget=QWidget()
        self.chunksize=int(self.gui.settings["dashboard_chunk_size"])
        self.n_cols=len(self.gui.settings["keys"].split(","))
        self.chunknumber=0
        self.grid=QGridLayout()
        self.time_label_buttons=[]
        self.buttonss=[]
        for i in range(self.chunksize):
            row=[]
            label_button=QPushButton(str(i+1) if i<=self.T else "")
            label_button.clicked.connect(self.make_button_press_function_t(i))
            label_button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
            label_button.setFixedWidth(30)
            self.time_label_buttons.append(label_button)
            self.grid.addWidget(label_button,i,0)
            for j in range(self.n_cols):
                button=QPushButton()
                button.clicked.connect(self.make_button_press_function_th(i,j))
                button.setFixedWidth(25)
                button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
                row.append(button)
                self.grid.addWidget(button,i,j+1)
            self.buttonss.append(row)        
        self.current_label_button=self.time_label_buttons[0]
        self.current_label_button.setStyleSheet("background-color : rgb(42,99,246); border-radius: 4px;")
        self.scrollwidget.setLayout(self.grid)
        
        self.scrollarea.setWidget(self.scrollwidget)
        self.scrollarea.setMinimumWidth(self.scrollarea.sizeHint().width()+self.scrollarea.verticalScrollBar().sizeHint().width())
        self.scrollarea.horizontalScrollBar().setEnabled(False)
        self.scrollarea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        maingrid=QGridLayout()
        topscrollarea=QScrollArea()
        topscrollarea.setFixedHeight(40)
        topscrollwidget=QWidget()
        topgrid=QGridLayout()
        button=QPushButton("")
        button.setFixedWidth(30)
        button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
        topgrid.addWidget(button,0,0)
        self.keys=self.gui.settings["keys"].split(",")
        self.assigned_colors=np.array([[int(val) for val in col.split(",")]  for col in self.gui.settings["keys_colors"].split(";")])
        for j in range(self.n_cols):
            button=QPushButton(self.keys[j])
            button.clicked.connect(self.make_button_press_function_h(j))
            button.setFixedWidth(25)
            button.setStyleSheet("background-color : rgb("+str(self.assigned_colors[j,0])+","+str(self.assigned_colors[j,1])+","+str(self.assigned_colors[j,2])+"); border-radius: 4px;")
            topgrid.addWidget(button,0,j+1)
        topscrollwidget.setLayout(topgrid)
        topscrollarea.setWidget(topscrollwidget)
        topscrollarea.setMinimumWidth(topscrollarea.sizeHint().width()+topscrollarea.verticalScrollBar().sizeHint().width())
        topscrollarea.horizontalScrollBar().setEnabled(False)
        topscrollarea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        maingrid.addWidget(topscrollarea,0,0)
        maingrid.addWidget(self.scrollarea,1,0)
        self.setLayout(maingrid)
    
    def make_button_press_function_t(self,i):
        def button_press_function():
            self.gui.respond("time_change",np.clip(self.chunksize*self.chunknumber+i+1,1,self.T))
        return button_press_function
        
    def make_button_press_function_h(self,j):
        def button_press_function():
            self.gui.respond("khighlight",j)
        return button_press_function
        
    def make_button_press_function_th(self,i,j):
        def button_press_function():
            self.gui.respond("time_change",np.clip(self.chunksize*self.chunknumber+i+1,1,self.T))
            self.gui.respond("khighlight",j)
        return button_press_function
    
    def update_time(self,time):
        chunknumber=(time-1)//self.chunksize
        if self.chunknumber!=chunknumber:
            for i in range(self.chunksize):
                t=self.chunksize*chunknumber+i+1
                self.time_label_buttons[i].setText(str(t) if t<=self.T else "")
            self.chunknumber=chunknumber

        self.current_label_button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
        self.current_label_button=self.time_label_buttons[(time-1)%self.chunksize]
        self.scrollarea.ensureWidgetVisible(self.current_label_button)
        self.current_label_button.setStyleSheet("background-color : rgb(42,99,246); border-radius: 4px;")
    
    def recolor(self,presence):
        chunknumber=(self.gui.time-1)//self.chunksize
        Ti=chunknumber*self.chunksize
        Tf=min((chunknumber+1)*self.chunksize,self.T)
        Tlen=Tf-Ti
        subpresence=np.zeros((Tlen,len(self.keys)))
        for i,key in enumerate(self.keys):
            i_point=self.gui.assigned_points[key]
            if i_point is None:
                subpresence[:,i]=-1
            else:
                subpresence[:,i]=presence[Ti:Tf,i_point]%2#0,2 for absences
                
        row=0
        for buttons in self.buttonss:
            if row==Tlen:
                row+=1
                break
            for j,button in enumerate(buttons):
                present=subpresence[row,j]
                if present==-1:
                    button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")
                elif present==0:
                    button.setStyleSheet("background-color : rgb(236,98,64); border-radius: 4px;")
                else:
                    button.setStyleSheet("background-color : rgb(42,99,246); border-radius: 4px;")
            row+=1
            
        for i in range(row,self.chunksize):
            for button in self.buttonss[i]:
                button.setStyleSheet("background-color : rgb(255,255,255); border-radius: 4px;")    
        
class PlotsTab(QTabWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.T=self.gui.data_info["T"]
        
        self.plot=Plot(self.gui)
        self.plot.setLabel("bottom","Time")
        self.set_axis_labels([])
        self.line=self.plot.addLine(x=1)
        self.t_vals=np.arange(1,self.T+1)
        
        self.addTab(self.plot,"Plot")
        self.tabBar().setTabTextColor(0,QtGui.QColor(0,0,0))
        
        self.choosergrid=QGridLayout()
        self.chooser = QListView()
        self.model = QtGui.QStandardItemModel()
        for series_name in self.gui.dataset.get_series_names():
            item = QtGui.QStandardItem(series_name)
            item.setCheckable(True)
            self.model.appendRow(item)
        self.chooser.setModel(self.model)
        self.choosergrid.addWidget(self.chooser,0,0)
        self.choosergrid.setRowStretch(0,10)
        self.plotbutton = QPushButton('Update')
        self.plotbutton.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-width: 40px; min-height: 20px")
        self.plotbutton.clicked.connect(self.update_plot)
        self.choosergrid.addWidget(self.plotbutton,1,0)
        self.choosergrid.setRowStretch(1,1)
        
        widget=QWidget()
        widget.setLayout(self.choosergrid)
        self.addTab(widget,"Choose Series")
        self.tabBar().setTabTextColor(1,QtGui.QColor(0,0,0))
    
    def set_axis_labels(self,names):
        stringaxis = pg.AxisItem(orientation='left')
        stringaxis.setTicks( [dict(zip(np.arange(len(names))+0.5,names)).items()] )
        self.plot.setAxisItems(axisItems = {'left': stringaxis})
    
    def normalize_series(self,series):
        valid=~np.isnan(series)
        if valid.sum()<2:
            return None
        else:
            m,M=series[valid].min(),series[valid].max()
            if not (M>m):
                return None
            series=(series-m)/(M-m)
            return series
    
    def update_plot(self):
        seriess={}
        series_labels=self.gui.dataset.get_series_labels()
        for i in range(self.model.rowCount()):
            item=self.model.item(i)
            if item.checkState()==Qt.Checked:
                name=item.text()
                seriess[series_labels[i]]=self.gui.dataset.get_data(name)
        if self.gui.signal is not None:
            for key,val in self.gui.assigned_points.items():
                if val is None:
                    continue
                seriess[key+"["+str(val)+"]"]=self.gui.signal[:,val]
        self.setCurrentIndex(0)
        self.plot.clear()
        self.set_axis_labels(list(seriess.keys()))
        self.line=self.plot.addLine(x=self.gui.time)
        base=0
        for name,series in seriess.items():
            series=self.normalize_series(series)
            if series is not None:
                self.plot.plot(x=self.t_vals,y=series+base,pen=pg.mkPen(width=1, color=(0,255,0)))
            base+=1
        self.plot.enableAutoRange()
        
    def update_time(self,time):
        self.line.setValue(time)

class Plot(pg.PlotWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui

class MinimapTab(pg.PlotWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        
class UtilityBar(QWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.grid=QGridLayout()
        
        save_button=QPushButton("Save")
        save_button.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-width: 40px; min-height: 20px")
        save_button.clicked.connect(lambda:self.gui.respond("save"))
        
        label_file=QLabel("File: "+self.gui.dataset.file_path)
        if "description" in self.gui.data_info.keys():
            description="Description: "+self.gui.data_info["description"]
        else:
            description="Description: "
        label_description=QLabel(description)
        
        self.grid.addWidget(save_button,0,0)
        self.grid.addWidget(label_file,0,1)
        self.grid.addWidget(label_description,0,2)

        self.setLayout(self.grid)

class LabeledSlider(QWidget):
    def __init__(self, minimum, maximum, interval=1, orientation=Qt.Horizontal,
            labels=None, parent=None):
        super(LabeledSlider, self).__init__(parent=parent)

        levels=list(range(minimum, maximum+interval, interval))
        #if levels[-1]>maximum:
        #    levels=levels[:-1]
        levels[-1]=maximum
        if labels is not None:
            if not isinstance(labels, (tuple, list)):
                raise Exception("<labels> is a list or tuple.")
            if len(labels) != len(levels):
                raise Exception("Size of <labels> doesn't match levels.")
            self.levels=list(zip(levels,labels))
        else:
            self.levels=list(zip(levels,map(str,levels)))

        if orientation==Qt.Horizontal:
            self.layout=QVBoxLayout(self)
        elif orientation==Qt.Vertical:
            self.layout=QHBoxLayout(self)
        else:
            raise Exception("<orientation> wrong.")

        # gives some space to print labels
        self.left_margin=10
        self.top_margin=10
        self.right_margin=10
        self.bottom_margin=10

        self.layout.setContentsMargins(self.left_margin,self.top_margin,
                self.right_margin,self.bottom_margin)

        self.sl=QSlider(orientation, self)
        self.sl.setMinimum(minimum)
        self.sl.setMaximum(maximum)
        self.sl.setValue(minimum)
        if orientation==Qt.Horizontal:
            self.sl.setTickPosition(QSlider.TicksBelow)
            self.sl.setMinimumWidth(300) # just to make it easier to read
        else:
            self.sl.setTickPosition(QSlider.TicksLeft)
            self.sl.setMinimumHeight(300) # just to make it easier to read
        self.sl.setTickInterval(interval)
        self.sl.setSingleStep(1)

        self.layout.addWidget(self.sl)

    def paintEvent(self, e):

        super(LabeledSlider,self).paintEvent(e)

        style=self.sl.style()
        painter=QPainter(self)
        st_slider=QStyleOptionSlider()
        st_slider.initFrom(self.sl)
        st_slider.orientation=self.sl.orientation()

        length=style.pixelMetric(QStyle.PM_SliderLength, st_slider, self.sl)
        available=style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self.sl)

        for v, v_str in self.levels:

            # get the size of the label
            rect=painter.drawText(QRect(), Qt.TextDontPrint, v_str)

            if self.sl.orientation()==Qt.Horizontal:
                # I assume the offset is half the length of slider, therefore
                # + length//2
                x_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
                        self.sl.maximum(), v, available)+length//2

                # left bound of the text = center - half of text width + L_margin
                left=x_loc-rect.width()//2+self.left_margin
                bottom=self.rect().bottom()

                # enlarge margins if clipping
                if v==self.sl.minimum():
                    if left<=0:
                        self.left_margin=rect.width()//2-x_loc
                    if self.bottom_margin<=rect.height():
                        self.bottom_margin=rect.height()

                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

                if v==self.sl.maximum() and rect.width()//2>=self.right_margin:
                    self.right_margin=rect.width()//2
                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

            else:
                y_loc=QStyle.sliderPositionFromValue(self.sl.minimum(),
                        self.sl.maximum(), v, available, upsideDown=True)

                bottom=y_loc+length//2+rect.height()//2+self.top_margin-3
                # there is a 3 px offset that I can't attribute to any metric

                left=self.left_margin-rect.width()
                if left<=0:
                    self.left_margin=rect.width()+2
                    self.layout.setContentsMargins(self.left_margin,
                            self.top_margin, self.right_margin,
                            self.bottom_margin)

            pos=QPoint(left, bottom)
            painter.drawText(pos, v_str)

        return

class TimeSliderWidget(LabeledSlider):
    def __init__(self,gui,T,n_labels):
        self.gui=gui
        if n_labels>T:
            n_labels=T
        super(TimeSliderWidget,self).__init__(minimum=1, maximum=T, interval=int(T/n_labels))
        self.sl.valueChanged.connect(lambda :self.gui.respond("time_change",self.sl.value()))

    def update_time(self,time):
        if self.sl.value()!=time:
            self.sl.blockSignals(True)
            self.sl.setValue(time)
            self.sl.blockSignals(False)
            
class GoToTimeWidget(QWidget):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.grid=QGridLayout()
        T=self.gui.data_info["T"]
        
        self.goto_button=QPushButton("Go To Time:")
        self.goto_button.setStyleSheet("background-color : rgb(93,177,130); border-radius: 4px; min-height: 20px")
        self.goto_button.clicked.connect(self.make_gototime_function())
        self.grid.addWidget(self.goto_button,0,0,1,1)
        self.tedit=QLineEdit("1")
        self.tedit.setValidator(QtGui.QIntValidator(1,T))
        self.grid.addWidget(self.tedit,0,1,1,1)
        self.setLayout(self.grid)
        
    def make_gototime_function(self):
        def gototime_function():
            t=self.tedit.text()
            try:
                t=int(t)
                return self.gui.respond("time_change",t)
            except Exception as ex:
                pass
                #print(ex)
        return gototime_function

        
