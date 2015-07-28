# Copyright JB Carruthers 2015
"""electric circuit diagramming using matplotlib

 Drawing
 -------

 diagram(Elements): draw the circuit Elements
 print_diagram(ImageFileName): export as an image file

 Elements
 --------

 voltage_source(A,B,...)
 current_source(A,B,...)

 opamp(Vmin,...)

 resistor(A,B,...)
 capacitor(A,B,...)
 inductor(A,B,...)
 device(A,B,...)

 wire(A,B,...)
 wires(NodeSeq,...)
 junction(A,...)
 ground(A,...)
 switch(A,B,...)

 arrow(A,B,)

 The first extra element is usually "elabel"
 ... represents matplotlib arguments like color, linewidth, etc.

 Labelling
 ---------
   vlabel draws a "+" and "-" and a voltage name
   elabel is for the element name, such as $R_1$

"""
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import matplotlib.patches as patch
from numpy import array,cos,sin,dot,arctan2,arange,pi
from matplotlib.pyplot import savefig 

__version__="0.1"

ko=r' k$\Omega$'
ohm='$\Omega$'

A,B,C=array((0.2,0.2)),array((0.8,0.8)),array((0.5,0.5))

box=array((0.2,0.2)),array((0.2,0.8)),array((0.8,0.8)),array((0.8,0.2))
box2=array((0.2,0.2)),array((0.5,0.2)),array((0.8,0.2)),array((0.2,0.6)),array((0.5,0.6)),array((0.8,0.6))
tall2=array((0.2,0.2)),array((0.2,0.5)),array((0.2,0.8)),array((0.6,0.2)),array((0.6,0.5)),array((0.6,0.8))
box3=array((0.1,0.2)),array((0.35,0.2)),array((0.60,0.2)),array((0.85,0.2)),array((0.1,0.6)),array((0.35,0.6)),array((0.6,0.6)),array((0.85,0.6))

#
#  E    F    G    H
#
#  A    B    C    D
Circuits=[]
Circuits.append(dict(zip('ABCDEFGH',((10,20),(35,20),(60,20),(85,20),
                                     (10,60),(35,60),(60,60),(85,60)))))
#  g  h  i 
#
#  d  e  f
#
#  a  b  c  
Circuits.append(dict(zip('adgbehcfi',(((10,10),(10,50),(10,90),(50,10),(50,50),(50,90),(90,10),(90,50),(90,90))))))
# w4 x4 y4 z4
# w3 x3 y3 z3
# w2 x2 y2 z2
# w1 x1 y1 z1
D={}
for i,x in enumerate('wxyz'):
  for j,y in enumerate('1234'):
    D[x+y]=(12.5+25*i,12.5+25*j)
Circuits.append(D)

# a3 b3 c3 d3 e3 f3
# a2 b2 c2 d2 e2 f2
# a1 b1 c1 d1 e1 f1
D={}
for i,x in enumerate('abcdef'):
  for j,y in enumerate('123'):
    D[x+y]=(10+17*i,20+30*j)
Circuits.append(D)

#
# 14 24 34 44 54
# 13 23 33 43 53 
# 12 22 32 42 52
# 11 21 31 41 51

D={}
for i,x in enumerate('12345'):
  for j,y in enumerate('1234'):
    D[x+y]=(12.5+20*i,12.5+25*j)
Circuits.append(D)


def map_letters(loccode):
  start,end=loccode
  #print(start,end)
  for circ in Circuits:
     if loccode[0] in circ:
         start,end = circ[start],circ[end]
         return (array(start)/100.0,array(end)/100.0)
  return loccode

def maploc(loccode):
  for circ in Circuits:
     if loccode in circ:
         return array(circ[loccode])/100.0
  return None


L=[]
for x in [0.1,0.4,0.7]:
 for y in [0.1,0.4,0.7]:
    L.append(array((x,y)))
thrbythr=tuple(L)

L=[]
for x in [0.1,0.4,0.7]:
 for y in [0.1,0.35,0.65,0.9]:
    L.append(array((x,y)))
thrbyfour=tuple(L)


class element(lines.Line2D):
    """the basis for other circuit elements"""
    def __init__(self,start_end='AB',mapletters=True,color='black',width=0,lw=1,vlabel=None,sep=0.2,elabel=None,*args,**kwargs):

      self.extras=[]   # extras are circuit drawing elements in addition to main "line"

      if mapletters:
        self.start,self.end = map_letters(start_end)
      else:
        self.start,self.end=start_end

      self.vec=self.end-self.start

      # elen is the size, th is the orientation, r is the transformation matrix
      self.elen=sum((self.vec)**2)**0.5  
      self.th=arctan2(self.vec[1],self.vec[0])
      self.r=self.elen*array([[cos(self.th),sin(self.th)],[-sin(self.th),cos(self.th)]])

      # voltage and element labels

      if elabel: # this is the element name or value

         the_angle=self.th*180/pi
         if abs(the_angle+90)<10:
             ha,va="left",'center'
         elif abs(the_angle-90)<10:
             ha,va="right",'center'
         else:
             ha,va="center",'bottom'
         elloc=self.start+dot([0.5,0.03+1.2*width],self.r) # .5,0 is middle, center
         self.extras.append(mtext.Text(elloc[0],elloc[1],elabel,va=va,ha=ha,size=20,color=color))   

      if vlabel: # this is a voltage designation
         vloc=self.start+dot([0.5,-0.15],self.r)
         the_angle=self.th*180/pi
         if abs(the_angle-90)<10:
             ha="left"
         elif abs(the_angle+90)<10:
             ha="right"
         else:
             ha="center"
         self.extras.append(mtext.Text(vloc[0],vloc[1],vlabel,va="center",ha=ha,size=20,color=color))
         ploc=self.start+dot([0.5+sep,-0.15],self.r)
         self.extras.append(lines.Line2D([ploc[0]],[ploc[1]],marker="+",mec=color,ms=12,*args, **kwargs))
         nloc=self.start+dot([0.5-sep,-0.15],self.r)
         self.extras.append(lines.Line2D([nloc[0]],[nloc[1]],marker="_",mec=color,ms=12,*args, **kwargs))

    #methods required for matplotlib drawing and updating

    def set_figure(self, figure):
      for k in self.extras:
            k.set_figure(figure)
      lines.Line2D.set_figure(self, figure)

    def set_axes(self, axes):
      for k in self.extras:
            k.set_axes(axes)
      lines.Line2D.set_axes(self, axes)

    def set_transform(self, transform):
      for k in self.extras:
            k.set_transform(transform)
      lines.Line2D.set_transform(self, transform)

    def draw(self, renderer):
      lines.Line2D.draw(self, renderer)
      for k in self.extras:
            k.draw(renderer)


class junction(patch.Circle):
   """junction(): make a wire junction (dot) at the location specified."""
   def __init__(self, loc='A',size=0.01,mapletters=True,color='black',lw=2,vlabel=None,elabel=None,*args, **kwargs):
      if mapletters:
        self.loc=maploc(loc)
      else:
        self.loc=loc
      patch.Circle.__init__(self,self.loc,size,color=color,lw=lw,alpha=1,*args, **kwargs)



class blank(element):
    """blank()"""
    def __init__(self, start_end="AB",elabel=None,vlabel=None, color="black",lw=2,sep=0.5,*args, **kwargs):
      element.__init__(self,start_end=start_end,color=color,lw=lw,vlabel=vlabel,elabel=elabel,sep=sep,*args,**kwargs)
      lines.Line2D.__init__(self,(self.start[0],self.end[0]),(self.start[1],self.end[1]),color=None,lw=0,*args, **kwargs)




class wire(element):
    """wire()"""
    def __init__(self, start_end="AB",mapletters=True,color='black',lw=2,*args, **kwargs):
      element.__init__(self,start_end=start_end,mapletters=mapletters,color=color,lw=lw,elabel=None,*args,**kwargs)
      lines.Line2D.__init__(self,(self.start[0],self.end[0]),(self.start[1],self.end[1]),color=color,lw=lw,*args, **kwargs)

class wires(element):
    """wires(): connect a list of nodes with a wire"""
    def __init__(self, nodelist=[A,B,C],mapletters=False,color='black',lw=2,*args, **kwargs):
      start,end=nodelist[:2]
      element.__init__(self,start_end=(start,end),mapletters=False,color=color,lw=lw,elabel=None,*args,**kwargs)
      lines.Line2D.__init__(self,(start[0],end[0]),(start[1],end[1]),color=color,lw=lw,*args, **kwargs)
      for start,end in zip(nodelist[1:-1],nodelist[2:]):
        self.extras.append(lines.Line2D((start[0],end[0]),(start[1],end[1]),color=color,lw=lw,*args, **kwargs))


class arrow(element):
    """arrow()"""
    def __init__(self, start_end="AB",elabel=None,mapletters=True,color='black',lw=2,width=0.1,*args, **kwargs):
      element.__init__(self,start_end=start_end,elabel=elabel,mapletters=mapletters,color=color,lw=lw,width=width,*args,**kwargs)
      lines.Line2D.__init__(self,(self.start[0],self.end[0]),(self.start[1],self.end[1]),color=color,lw=lw,*args, **kwargs)

      s=array([(0.47,-0.06),(0.53,0),(0.47,0.06)])
      vp=self.start+dot(s,self.r)
      self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))

class ground(element):
    """ground()"""
    def __init__(self, loc=C,mapletters=True,size=1,color='black',lw=2,*args, **kwargs):
      element.__init__(self,start_end=(loc,loc),mapletters=mapletters,color=color,lw=lw,*args,**kwargs)
      
      lines.Line2D.__init__(self,(self.start[0]-0.03*size,self.start[0]+0.03*size),(self.start[1],self.start[1]),
                                 color=color,lw=lw,*args, **kwargs)

      self.extras.append(lines.Line2D((self.start[0]-0.02*size,self.start[0]+0.02*size),
          (self.start[1]-0.007*size,self.start[1]-0.007*size),color=color,lw=lw,*args,**kwargs))

      self.extras.append(lines.Line2D((self.start[0]-0.01*size,self.start[0]+0.01*size),
          (self.start[1]-0.014*size,self.start[1]-0.014*size),color=color,lw=lw,*args,**kwargs))


class switch(element):
    """switch()"""
    def __init__(self, start_end="AB",dir='open',elabel=None,vlabel=None,color='black',lw=2,width=0.2,*args, **kwargs):
      element.__init__(self,start_end=start_end,width=width,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

      #standard switch goes left to right.
      self.v=array([(0,0),(0.3,0),(0.7,0.2)])
      vp=self.start+dot(self.v,self.r)
      lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)
      # Finish the rest of the switch
      self.v2=array([(0.75,0),(1,0)])

      v2p=self.start+dot(self.v2,self.r)
      self.extras.append(lines.Line2D(v2p[:,0],v2p[:,1],color=color,lw=lw,*args,**kwargs))
    
      arc_center=self.start+dot(array((0.3,0)),self.r)
      self.extras.append(patch.Arc(arc_center,0.5*self.elen,0.5*self.elen, # width, height of arc
            theta1=180*self.th/pi,theta2=60+self.th*180/pi,color=color,lw=lw,*args,**kwargs))
      
      if dir=='close':
        s=array([(0.52,0.03),(0.55,0),(0.58,0.03)])
        vp=self.start+dot(s,self.r)
        self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))
      elif dir=='open':
        s=array([(0.425,0.17),(0.425,0.22),(0.47,0.22)])
        vp=self.start+dot(s,self.r)
        self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))



class resistor(element):
    """resistor()"""
    def __init__(self, start_end='AB',elabel=None,vlabel=None,mapletters=True,color='black',lw=2,width=0.05,*args, **kwargs):
      element.__init__(self,start_end=start_end,mapletters=mapletters,width=width,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

      #standard resistor goes left to right.
      self.v=array([(0,0),(0.32,0),(0.35,-.05),(.41,0.05),(.47,-0.05),(.53,0.05),(.59,-0.05),(.65,0.05),(.68,0),(1,0)])
      vp=self.start+dot(self.v,self.r)
      lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)

class device(element):
    """device()"""
    def __init__(self, start_end='AB',elabel=None,vlabel=None,mapletters=True,color='black',width=0.12,lw=2,*args, **kwargs):
       element.__init__(self,start_end=start_end,width=width,mapletters=mapletters,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

       wid=0.06
       devlines=[ [(0,0),(0.32,0)],
                  [(0.68,0),(1.0,0)],
                  [(0.32,-wid),(0.32,wid),(0.68,wid),(0.68,-wid),(0.32,-wid)]]
                  

       vp=self.start+dot(array(devlines[0]),self.r)
       lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)
       for l in devlines[1:]:
         vp=self.start+dot(array(l),self.r)
         self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))




class opamp(element):
    """opamp()"""
    plusminus_spread=0.28 # controls spread of plus and minus terminals
    standard_size=0.3 # standard size 
    
    def __init__(self, nodeloc=C,mapletters=False,node='vminus',direction='right',size=1.0,color='black',lw=2,vlabel=None,elabel=None,spread=1,*args, **kwargs):
      #start is middle between v+ and v-. end is vout.
     

      if node=='vminus':
        Offset={'right':(0,-1),'left':(0,1),'up':(1,0),'down':(-1,0)}
        Nose={'right':(1,0),'left':(-1,0),'up':(0,1),'down':(0,-1)}
        start = nodeloc + array(Offset[direction])*size*opamp.standard_size*opamp.plusminus_spread
        end = start + array(Nose[direction])*size*opamp.standard_size
      else:
        raise NotImplementedError('specify vminus')

      self.direction=direction
      self.spread=opamp.plusminus_spread*spread

      element.__init__(self,start_end=(start,end),mapletters=mapletters,width=0.3,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

      #standard opamp
      self.v=array([(0,0),(0,0.5),(1,0),(0,-0.5),(0,0)])
      vp=self.start+dot(self.v,self.r)
      lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)

      for (mark,loc) in [ ("+",(0.12,-self.spread)),("_",(0.12,+self.spread))]:
          markloc=self.start+dot(loc,self.r)
          self.extras.append(lines.Line2D([markloc[0]],[markloc[1]],marker=mark,mec=color,ms=20*size,*args, **kwargs))
    
    def getnodes(self):
         "find the locations of the three connection points: +,-,out"
         nodelocs={}
         for (mark,loc) in [ ("+",(0.0,-self.spread)),("-",(0.0,self.spread)),('out',(1.0,0))]:
             nodelocs[mark]=self.start+dot(loc,self.r)

         return nodelocs['+'],nodelocs['-'],nodelocs['out']


class inductor(element):
     """inductor()"""
     def __init__(self, start_end='AB',elabel=None,vlabel=None,color='black',lw=2,width=0.15,*args, **kwargs):
       element.__init__(self,start_end=start_end,width=0.1,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)
       #standard inductor goes left to right.

       elines=[ [(0,0),(0.3,0)] , [(0.7,0),(1,0)] ]

       vp=self.start+dot(array(elines[0]),self.r)
       lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)

       vp=self.start+dot(array(elines[1]),self.r)
       self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))
      
       for cen in arange(0.35,0.65,0.1):
         vp=self.start+dot(array([cen,0]),self.r)
         self.extras.append(patch.Arc(vp,0.1*self.elen,0.1*self.elen, # width, height of arc
            theta1=180*self.th/pi,theta2=180+self.th*180/pi,color=color,lw=lw,*args,**kwargs))



class capacitor(element):
     """capacitor()"""
     def __init__(self, start_end="AB",elabel=None,vlabel=None,color='black',lw=2,width=0.15,*args, **kwargs):
       element.__init__(self,start_end=start_end,width=width,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

       #standard capacitor goes left to right.
       capgap=0.05
       capwidth=0.08
       caplines=[ [(0,0),(0.5-capgap,0)],
                  [(0.5+capgap,0),(1,0)],
                  [(0.5-capgap,-capwidth),(0.5-capgap,capwidth)],
                  [(0.5+capgap,-capwidth),(0.5+capgap,capwidth)] ]

       vp=self.start+dot(array(caplines[0]),self.r)
       lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)
       for l in caplines[1:]:
         vp=self.start+dot(array(l),self.r)
         self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))



class voltage_source(element):
     """voltage_source()"""
     def __init__(self, start_end="AB",elabel=None,vlabel=None,mapletters=True,color='black',lw=2,width=0.12,*args, **kwargs):
       element.__init__(self,start_end=start_end,width=width,mapletters=mapletters,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

       #standard goes left to right.
       vlines=[ [(0,0),(0.38,0)],
                [(0.62,0),(1,0)],
                [(0.46,-0.03),(0.46,0.03)],
                [(0.55,-0.03),(0.55,0.03)],
                [(0.58,0),(0.52,0)] ]
                
       vp=self.start+dot(array(vlines[0]),self.r)
       lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)
       for l in vlines:
         vp=self.start+dot(array(l),self.r)
         self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))
       
       self.extras.append(patch.Circle(self.start+self.vec/2,0.12*self.elen,
               ec=color,fc='none',lw=lw,*args,**kwargs))

class current_source(element):
     """current_source()"""
     def __init__(self, start_end="AB",elabel=None,vlabel=None,color='black',lw=2,width=0.13,*args, **kwargs):
       element.__init__(self,start_end=start_end,width=width,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

       #standard goes left to right.
       vlines=[ [(0,0),(0.38,0)],
                [(0.62,0),(1,0)]]
                
       vp=self.start+dot(array(vlines[0]),self.r)
       lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)
       for l in vlines:
         vp=self.start+dot(array(l),self.r)
         self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))
       
       self.extras.append(patch.Circle(self.start+self.vec/2,0.12*self.elen,ec=color,fc='none',lw=lw,*args,**kwargs))

       x,y=self.start+0.45*self.vec
       dx,dy=0.1*self.vec
       self.extras.append(patch.Arrow(x,y,dx,dy,
               ec=color,fc='none',width=0.1*self.elen,*args,**kwargs))


class dep_voltage_source(element):
     """voltage_source()"""
     def __init__(self, start_end="AB",elabel=None,vlabel=None,color='black',lw=2,width=0.12,*args, **kwargs):
       element.__init__(self,start_end=start_end,width=width,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

       #standard goes left to right.
       ds=0.14
       devlines=[ [(0,0),(0.5-ds,0)],
                  [(0.5-ds,0),(0.5,ds)],
                  [(0.5,ds),(0.5+ds,0)],
                  [(0.5+ds,0),(0.5,-ds)],
                  [(0.5,-ds),(0.5-ds,0)],
                  [(0.5+ds,0),(1,0)],
                  [(0.46,-0.03),(0.46,0.03)],
                  [(0.55,-0.03),(0.55,0.03)],
                  [(0.58,0),(0.52,0)],
                  ]

       vp=self.start+dot(array(devlines[0]),self.r)
       lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)
       for l in devlines[1:]:
         vp=self.start+dot(array(l),self.r)
         self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))        
        

class dep_current_source(element):
     """dependent current source"""
     def __init__(self, start_end="AB",elabel=None,vlabel=None,color='black',lw=2,width=0.13,*args, **kwargs):
       element.__init__(self,start_end=start_end,width=width,color=color,lw=lw,vlabel=vlabel,elabel=elabel,*args,**kwargs)

       #standard goes left to right.
       ds=0.14
       devlines=[ [(0,0),(0.5-ds,0)],
                  [(0.5-ds,0),(0.5,ds)],
                  [(0.5,ds),(0.5+ds,0)],
                  [(0.5+ds,0),(0.5,-ds)],
                  [(0.5,-ds),(0.5-ds,0)],
                  [(0.5+ds,0),(1,0)],
                  ]

       vp=self.start+dot(array(devlines[0]),self.r)
       lines.Line2D.__init__(self,vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs)
       for l in devlines[1:]:
         vp=self.start+dot(array(l),self.r)
         self.extras.append(lines.Line2D(vp[:,0],vp[:,1],color=color,lw=lw,*args, **kwargs))        


       x,y=self.start+0.45*self.vec
       dx,dy=0.1*self.vec
       self.extras.append(patch.Arrow(x,y,dx,dy,
               ec=color,fc='none',width=0.1*self.elen,*args,**kwargs))


def diagram(Elements):
   """diagram(Elements) creates a new circuit diagram figure and draws each item in Elements"""
   plt.figure()
   ax=plt.axes([0,0,1,1],frameon=False,xticks=[],yticks=[],aspect=1)
   for e in Elements:
     if type(e) in [junction]:
      ax.add_patch(e)
     else:
      ax.add_line(e)

   return ax

import os
def make_it(s):
    """get a cropped pdf from the current figure"""
    savefig(s+'.pdf')
    plt.close()
    os.system('pdfcrop {}.pdf'.format(s))
    os.rename(s+'-crop.pdf',s+'.pdf')

def crop(fname,suffix='c'):
  from PIL import Image
  im=Image.open(fname)
  box=im.getbbox()
  #print(box)
  cr=im.crop(box)
  first,second=fname.split('.')
  cr.save('{}{}.{}'.format(first,suffix,second))

def print_diagram(fname,tra=True,**kwargs):
  #print(kwargs)
  savefig(fname,transparent=tra,**kwargs)
  plt.close()
  crop(fname,'')

def make_figure_file(fname,**kwargs):
  print(fname)
  if fname.endswith('pdf'):
    make_it(fname[:-4])
  else:
    print_diagram(fname,**kwargs)


def rs(g):
    """resistor string"""
    if g==1:
        return 'R'
    else:
        return "{}R".format(g)

def par(r):
    return 1/sum(1/x for x in r) 


CS=current_source
R=resistor
L=inductor
C=capacitor
W=wire
Ws=wires
VS=voltage_source
J=junction


