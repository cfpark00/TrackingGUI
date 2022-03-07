from PyQt5.QtWidgets import *#QApplication,QMainWindow,QWidget,QMessageBox
from src.CustomWidgets import *
from src.Dataset import *
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui

import threading
import time
import sys
import pyqtgraph as pg
import numpy as np
import scipy.interpolate as sintp

class GUI():
    def __init__(self,file_path,settings):
        self.settings=settings
        self.dataset=Dataset(file_path)

        self.close=False
        self.assigned_points={}
        self.key_index={}
        i=0
        self.keys=self.settings["keys"].split(",")
        for key in self.keys:
            self.assigned_points[key]=None
            self.key_index[key]=i
            i+=1
        self.click_distance=float(self.settings["click_distance"])

        self.dataset.open()
        self.points=self.dataset.get_points()
        self.data_info=self.dataset.get_data_info()

        self.time=1
        self.z_float=0
        self.z=0
        self.scroll_per_z=float(self.settings["scroll_per_z"])
        self.T=self.data_info["T"]
        self.Tstr="/"+str(self.T)+" "
        self.D=self.data_info["D"]
        self.Dstr="/"+str(self.D)+" "

        self.W,self.H=self.data_info["W"],self.data_info["H"]

        self.helper=None
        self.signal=None
        self.highlighted=0
        self.follow=False

        self.app = QApplication(sys.argv)
        self.app.setWindowIcon(QtGui.QIcon('src/whale.png'))

        screen = self.app.primaryScreen()
        size = screen.size()
        self.settings["screen_w"]=size.width()
        self.settings["screen_h"]=size.height()
        self.fps=float(self.settings["fps"])

        self.win = Window(self)
        self.win.show()


    def start(self):
        self.timer=QTimer()
        self.timer.timeout.connect(lambda:self.respond("update_data"))
        self.last_update_t=time.time()
        self.respond("timer_start")
        self.app.exec()
        self.respond("timer_stop")
        self.dataset.close()

    def respond(self,key,val=None):
        if key=="close":
            self.close=True
        elif key=="update_assignments":
            self.assigned_points[val[0]]=val[1]
            self.update_assigned()
        elif key=="dtime_change":
            self.time=np.clip(self.time+val,1,self.data_info["T"])
            self.update_time()
        elif key=="time_change":
            self.time=val
            self.update_time()
        elif key=="update_data":
            if self.close:
                return
            if self.z==-1:
                data={"image":self.dataset.get_frame(self.time-1).max(3)}
            else:
                data={"image":self.dataset.get_frame_z(self.time-1,self.z)}
            coords=self.points[self.time-1].copy()
            pt_type=np.full(coords.shape[0],-1)
            if self.helper is not None:
                nvalids=np.isnan(coords[:,0])
                coords[nvalids]=self.helper[self.time-1,nvalids]
                pt_type[nvalids]=-2

            for key,index in self.assigned_points.items():
                if index is not None:
                    pt_type[index]=self.key_index[key]
            pt_type[self.highlighted]=-3

            points=np.concatenate([coords,pt_type[:,None]],axis=1)
            self.plotpoints=points
            data["points"]=points

            label="Frame: "+str(self.time)+self.Tstr
            label+=" z: "+(str(self.z)+self.Dstr) if self.z>=0 else " z: max "
            dt=time.time()-self.last_update_t
            label+="fps: "+str(np.round(1/dt,1))
            data["label"]=label
            data["z"]=self.z
            self.win.figurewidget.update_data(data)
            self.update_presence()
            self.last_update_t=time.time()

        elif key=="fig_keypress":
            kkey=val[0]
            if kkey not in self.assigned_points.keys():
                if kkey=="d":
                    if self.z<0:
                        return
                    coords=self.plotpoints[:,:3]
                    valid=~np.isnan(coords[:,0])
                    if np.sum(valid)==0:
                        return
                    indices=np.nonzero(valid)[0]
                    dists=np.linalg.norm(coords[valid]-np.array([val[1],val[2],self.z])[None,:],axis=1)
                    am=np.argmin(dists)
                    if dists[am]<self.click_distance:
                        self.respond("delete_single",indices[am])
                    return
                else:
                    print("key press",kkey)
                    return
            coord=np.array([val[1],val[2],self.z]).astype(np.float32)
            if self.assigned_points[kkey] is not None:
                if (-0.5<coord[0]<(self.W-0.5)) and (-0.5<coord[1]<(self.H-0.5)) and (-0.5<coord[2]<(self.D-0.5)):
                    i_point=self.assigned_points[kkey]
                    self.points[self.time-1,i_point]=coord
        elif key=="delete_single":
            self.points[self.time-1,val,:]=np.nan
        elif key=="fig_click":
            if val[0]==1:
                coords=self.plotpoints[:,:3]
                valid=~np.isnan(coords[:,0])
                if np.sum(valid)==0:
                    return
                indices=np.nonzero(valid)[0]
                dists=np.linalg.norm(coords[valid]-np.array([val[1],val[2],self.z])[None,:],axis=1)
                am=np.argmin(dists)
                if dists[am]<self.click_distance:
                    self.respond("highlight",indices[am])
            else:
                print("coord:",val[1],val[2],self.z)
        elif key=="fig_scroll":
            dz=val/self.scroll_per_z
            self.z_float=np.clip(self.z_float+dz,-1.5,self.D-0.5)
            self.update_z()
        elif key=="view_change":
            self.win.figurewidget.update_params(val)
        elif key=="highlight":
            if self.highlighted==val:
                self.highlighted=0
            else:
                self.highlighted=val
        elif key=="khighlight":
            i_point=self.assigned_points[self.keys[val]]
            if i_point is None:
                return
            else:
                self.respond("highlight",i_point)
        elif key=="load_helper":
            if val=="":
                self.helper=None
            else:
                self.helper=self.dataset.get_helper(val)
        elif key=="renew_helpers":
            self.helper=None
            self.win.annotate_tab.renew_helper_list()
        elif key=="load_signal":
            if val=="":
                self.signal=None
            else:
                self.signal=self.dataset.get_signal(val)
                self.win.plots_tab.update_plot()
        elif key=="renew_signals":
            self.signal=None
            self.win.analysis_tab.renew_signal_list()
            self.win.plots_tab.update_plot()
        elif key=="follow":
            self.follow=val
        elif key=="save":
            self.dataset.set_points(self.points)
            print("Saved")
        elif key=="timer_stop":
            self.timer.stop()
        elif key=="timer_start":
            self.timer.start(1000/self.fps)
        elif key=="lin_intp":
            ti,tf=val
            if ti>tf:
                return
            locs=self.points[ti-1:tf,self.highlighted]
            if np.sum(~np.isnan(locs[0]))<2:
                print("At least 2 annotations needed")
                return
            isgt=~np.isnan(self.points[ti-1:tf,self.highlighted,0])
            gt_inds=np.nonzero(isgt)[0]
            not_gt_inds=np.nonzero(~isgt)[0]
            intp_func=sintp.interp1d(gt_inds, locs[gt_inds], kind='linear', axis=0, bounds_error=False, fill_value=np.nan)
            locs_res=intp_func(not_gt_inds)
            self.points[not_gt_inds+ti-1,self.highlighted,:]=locs_res
            print("Linearly Interpolated",self.highlighted,"from",ti,"to",tf)
        elif key=="delete":
            ti,tf=val
            if ti>tf:
                return
            self.points[ti-1:tf,self.highlighted,:]=np.nan
            print("Deleted",self.highlighted,"from",ti,"to",tf)
        elif key=="approve":
            if self.helper is None:
                return
            ti,tf=val
            if ti>tf:
                return
            locs=self.helper[:,self.highlighted]
            not_gt=np.nonzero(np.isnan(self.points[ti-1:tf,self.highlighted,0]))[0]+ti-1
            self.points[not_gt,self.highlighted,:]=locs[not_gt]
            print("Approved",self.highlighted,"of helper",self.win.annotate_tab.get_current_helper_name(),"from",ti,"to",tf)
        elif key=="transpose":
            self.win.figurewidget.set_transpose(val)
        else:
            print(key,val)

    def update_time(self):
        if self.follow:
            if self.highlighted!=0:
                ztarg=self.points[self.time-1,self.highlighted,2]
                if not np.isnan(ztarg):
                    self.z_float=ztarg
                elif self.helper is not None:
                    ztarg_helper=self.helper[self.time-1,self.highlighted,2]
                    if not np.isnan(ztarg_helper):
                        self.z_float=ztarg_helper
            self.update_z()
        self.win.timesliderwidget.update_time()
        self.win.dashboard_tab.update_time()
        self.win.plots_tab.update_time()

    def update_presence(self):
        #0 for absent, 1 for present, 2 for highlighted absent, 3 for highlighted present
        presence=(~np.isnan(self.points[:,:,0])).astype(np.int16)
        presence[:,self.highlighted]+=2
        self.win.pointbarwidget.recolor(presence[self.time-1])
        self.win.dashboard_tab.recolor(presence)
        self.win.annotate_tab.highlight(self.highlighted)

    def update_z(self):
        self.z=np.clip(int(self.z_float+0.5),-1,self.D-1)

    def update_assigned(self):
        self.win.pointbarwidget.update_assigned()
        self.win.plots_tab.update_plot()

class Window(QMainWindow):
    def __init__(self,gui):
        super().__init__()
        self.gui=gui
        self.setWindowTitle("Tracking")
        self.resize(self.gui.settings["screen_w"]*4//5, self.gui.settings["screen_h"]*4//5)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        if True:
            maingrid=QGridLayout()
            if True:
                topgrid=QGridLayout()
                if True:
                    self.topbar=UtilityBar(self.gui)
                    topgrid.addWidget(self.topbar)

                leftgrid=QGridLayout()
                if True:
                    self.pointbarwidget=PointBarWidget(self.gui)
                    self.figurewidget=Annotated3DFig(self.gui)
                    self.timesliderwidget=TimeSliderWidget(self.gui,self.gui.data_info["T"],10)
                    leftgrid.addWidget(self.pointbarwidget,0,0)
                    leftgrid.addWidget(self.figurewidget,1,0)
                    leftgrid.addWidget(self.timesliderwidget,2,0)
                    leftgrid.setRowStretch(0,0)
                    leftgrid.setRowStretch(1,1)
                    leftgrid.setRowStretch(2,0)

                rightgrid=QGridLayout()
                if True:
                    self.plotstab=QTabWidget()
                    #self.plotstab.setFixedSize(self.gui.settings["screen_w"]//5,self.gui.settings["screen_h"]//3)
                    if True:
                        self.dashboard_tab=DashboardTab(self.gui)
                        self.plotstab.setFixedSize(self.dashboard_tab.sizeHint().width(),self.gui.settings["screen_h"]//3)
                        self.plots_tab=PlotsTab(self.gui)
                        self.minimap_tab=MinimapTab(self.gui)

                        self.plotstab.addTab(self.dashboard_tab,"Dashboard")
                        self.plotstab.tabBar().setTabTextColor(0,QtGui.QColor(0,0,0))
                        self.plotstab.addTab(self.plots_tab,"Plots")
                        self.plotstab.tabBar().setTabTextColor(1,QtGui.QColor(0,0,0))
                        self.plotstab.addTab(self.minimap_tab,"Minimap")
                        self.plotstab.tabBar().setTabTextColor(2,QtGui.QColor(0,0,0))

                    self.controlstab=QTabWidget()
                    if True:
                        self.view_tab=ViewTab(self.gui)
                        self.annotate_tab=AnnotateTab(self.gui)
                        self.track_tab=TrackTab(self.gui)
                        self.analysis_tab=AnalysisTab(self.gui)

                        self.controlstab.addTab(self.view_tab,"View")
                        self.controlstab.tabBar().setTabTextColor(0,QtGui.QColor(0,0,0))
                        self.controlstab.addTab(self.annotate_tab,"Annotate")
                        self.controlstab.tabBar().setTabTextColor(1,QtGui.QColor(0,0,0))
                        self.controlstab.addTab(self.track_tab,"Track")
                        self.controlstab.tabBar().setTabTextColor(2,QtGui.QColor(0,0,0))
                        self.controlstab.addTab(self.analysis_tab,"Analysis")
                        self.controlstab.tabBar().setTabTextColor(3,QtGui.QColor(0,0,0))

                    self.gototimewidget=GoToTimeWidget(self.gui)
                    rightgrid.addWidget(self.plotstab,0,0)
                    rightgrid.addWidget(self.controlstab,1,0)
                    rightgrid.addWidget(self.gototimewidget,2,0)
                maingrid.addLayout(topgrid,0,0)
                maingrid.addLayout(leftgrid,1,0)
                maingrid.addLayout(rightgrid,1,1)
                maingrid.setColumnStretch(0,5)
                maingrid.setColumnStretch(1,1)
            self.centralWidget.setLayout(maingrid)

            #not widget dependent, global
            tkeys=self.gui.settings["tkeys"].split(",")
            tdeltas=[int(val) for val in self.gui.settings["tdeltas"].split(",")]
            for tkey,tdelta in zip(tkeys,tdeltas):
                short=QShortcut(QKeySequence(tkey), self)
                short.activated.connect(self.get_dtime_change_func(tdelta))

    def get_dtime_change_func(self,tdelta):
        def func():
            self.gui.respond("dtime_change",tdelta)
        return func

    def closeEvent(self,event):
        msgbox = QMessageBox(QMessageBox.Question, "", "Save and Quit?")
        msgbox.addButton(QMessageBox.Yes)
        msgbox.addButton(QMessageBox.No)
        msgbox.addButton(QMessageBox.Cancel)
        msgbox.setDefaultButton(QMessageBox.Cancel)
        reply = msgbox.exec()
        if reply == QMessageBox.Yes:
            self.gui.respond("save")
            self.gui.respond("close")
            event.accept()
        elif reply==QMessageBox.No:
            self.gui.respond("close")
            event.accept()
        else:
            event.ignore()
