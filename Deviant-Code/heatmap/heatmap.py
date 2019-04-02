import csv
import math
import unicodedata

from  scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.switch_backend('agg')
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from util.matrixProcessing import getInnerMatrix,getCompleteMatrix

def show_values(pc, fmt="%d",highlight=None,fontsize=60, **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        if highlight is None:
            ax.text(x, y, fmt % (value*100)+ ' %', ha="center", va="center", color=color,fontsize=fontsize, fontweight='bold', **kw)
        else:
            if highlight[int(y)][int(x)]:
                ax.text(x, y, fmt % (value*100)+ ' %', ha="center", va="center", color=color,fontsize=fontsize, fontweight='bold', **kw)

def generateHeatMap(dataset,destination,color='RdYlGn',vmin=None,vmax=None,organize=False,title=None,showvalues_text=False,only_heatmap=True,highlight=None):
    '''
    Choosed RdYlGn
    @note : color = 'RdYlGn' or 'RdYlGn_r' summer_r
    '''
    #dataset = readCSV(source, delimiter=',')
    innerMatrix,rower,header=getInnerMatrix(dataset)
    completeMatrix=dataset
    rower=[unicodedata.normalize('NFD', unicode(str(rower[k]),'iso-8859-1')).encode('ascii', 'ignore') for k in sorted(rower)]
    header=[unicodedata.normalize('NFD', unicode(str(header[k]),'iso-8859-1')).encode('ascii', 'ignore')  for k in sorted(header)]  
    rower_inv={v:k for k,v in enumerate(rower)}
    header_inv={v:k for k,v in enumerate(header)}

    nba=pd.DataFrame(innerMatrix,index=rower,columns=header,dtype=float)
    matrixSimilairty=nba.as_matrix()
    header_new=header[:]
    rower_new=rower[:]
    
    ##########################################################################
    if organize :
        matrix=nba.as_matrix()
        matrix=[[1-x for x in row] for row in matrix]
        isSquare=True
        if len(matrix)<>len(matrix[0]):
            isSquare=False
        if isSquare:
            for index,row in enumerate(matrix) :
                for column,val in enumerate(row) :
                    if not innerMatrix[column][index] == matrix[index][column] :
                        if (math.isnan(matrix[index][column])) :
                            matrix[index][column]=1.
                        #print innerMatrix[index][column],'-',innerMatrix[column][index] ##ERREURE d'ARRONDIE
                        else :
                            matrix[index][column]=matrix[column][index] ##ERREURE d'ARRONDIE
                    if index==column:
                        matrix[index][column]=0.
        
        
        
        
            distArray = ssd.squareform(matrix)    
            linkageMatrix=linkage(distArray, 'average')
            
            cuttreeclusters=sorted([(i,t) for (i,t) in enumerate(hierarchy.fcluster(linkageMatrix,0.2,'distance'))],key=lambda x : x[1])
            clusters={}
            for i,c in cuttreeclusters:
                if not clusters.has_key(c):
                    clusters[c]=[]
                clusters[c].append(i)
            #print clusters
                
            cuttreeclustersSorted=[]
             
            pairs=[]
            for i in range(len(matrix)):
                row = matrix[i]
                for j in range(len(row)):
                    pairs.append((i,j,row[j]))
            pairs=sorted(pairs,key=lambda x : x[2])
            visited=[]
            for c,c_elems in clusters.iteritems():
                for i,j,d in pairs:
                    if i in c_elems and j in c_elems:
                        if i not in visited:
                            visited.append(i)
                            cuttreeclustersSorted.append((i,c))
                        if j not in visited:
                            visited.append(j)
                            cuttreeclustersSorted.append((j,c))
            rower_new=[rower[t[0]] for t in cuttreeclustersSorted]
            header_new=[header[t[0]] for t in cuttreeclustersSorted]

            new_inner_matrix=[[innerMatrix[rower_inv[rowVal]][header_inv[headVal]] for headVal in header_new] for rowVal in rower_new]
            completeMatrix=getCompleteMatrix(new_inner_matrix,{xx:yy for xx,yy in enumerate(rower_new)},{xx:yy for xx,yy in enumerate(header_new)})

            nba=nba.reindex(index=rower_new,columns=header_new)
        else:
            cuttreeclustersSorted=[]
            pairs=[]
            for i in range(len(matrix)):
                row = matrix[i]
                for j in range(len(row)):
                    pairs.append((i,sum(row)))
            pairs=sorted(pairs,key=lambda x : x[1])       
            visited=[]     
            for i,d in pairs:
                if i not in visited:
                    visited.append(i)
                    cuttreeclustersSorted.append((i,0))         
        
            rower_new=[rower[t[0]] for t in cuttreeclustersSorted]
            header_new=header[:]
            nba=nba.reindex(index=rower_new,columns=header_new)
        
    
    
    ##########################################################################
    
    
    fig, ax = plt.subplots()
    #fig.gca().set_position((.4, .4, .8, .8))
    masked_array = np.ma.array (nba, mask=np.isnan(nba))
    heatmap = ax.pcolor(masked_array, cmap=plt.cm.get_cmap(name=color), alpha=1,vmax=vmax,vmin=vmin)
    
    
    
    if showvalues_text :
        show_values(heatmap,highlight=highlight,fontsize=45)

    if highlight is not None:
        for i in range(len(highlight)):
            for j in range(len(highlight[i])):
                if highlight[i][j]:
                    #print i,j
                    #ax.add_patch(Rectangle((j, i), 1, 1, fill=True, edgecolor='black',facecolor='black', lw=1,alpha=0.8))
                    #ax.add_patch(Rectangle((j, i), 1, 1, fill=True, edgecolor='black',facecolor='blue', lw=3,alpha=0.35))

                    #ax.add_patch(Rectangle((j, i), 1, 1, fill=True, edgecolor='black',facecolor='white', lw=0.01,alpha=0.50)) #nothighlighed = FALSE
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=True, edgecolor='black',facecolor='black', lw=2,alpha=0.25)) #Beautiful

    fig = plt.gcf()
    
    if not only_heatmap:
        fig.subplots_adjust(bottom=0.2,top=0.87,right=1.)
    fig.set_size_inches(40, 40) #40, 40
    
    ax.set_frame_on(False)
    ax.set_yticks(np.arange(nba.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(nba.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    if not only_heatmap:
        fig.suptitle(title, fontsize=45,x =0.5,y = 0.1)#fontweight='bold'
    xlabels= header_new#header[:]
    ylabels=rower_new#rower
    
    if True:
        ax.set_xticklabels(xlabels, minor=False,fontsize=15)
        ax.set_yticklabels(ylabels, minor=False,fontsize=15)
        print xlabels
        
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.colorbar(heatmap)
    ax.grid(False)

    ax = plt.gca()
    
    
    
    #ax.set_position((.1, .3, .8, .6))
    

    # if not only_heatmap:
    #     plt.figtext(0.12, .1,'Details about :\n * the pattern \n * dossiers \n * compared MEPs',fontsize=40)
    
    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    if not only_heatmap:
        plt.colorbar(heatmap)
    #fig.tight_layout()
    if only_heatmap: 
        plt.tight_layout()
    plt.savefig(destination,dpi=300)
    #print 'hello !!!!!!!'
    fig.clf()
    plt.clf()
    plt.gcf().clear()
    plt.cla()
    plt.close('all')
    return completeMatrix