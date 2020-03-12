from sympy.utilities.iterables import multiset_permutations
import numpy as np

class Polytope:

    def __init__(self, verts, picardNum, imbeddingDim):
        self.verts = verts
        self.imbeddingDim = imbeddingDim
        self.picardNum = picardNum
        self.translations = []

    def generateData(self):
        temp = []
        for p in multiset_permutations(self.verts):
            temp.append(p)
        self.verts = temp
        self.translations.append(self.verts)
        #### Construct transformations of base vectors and place each in a list wrapped in a master list
        ## Reflection across y = x
        transformation = []
        for p in self.verts:
            vect = []
            for v in p:
                x = v[1]
                y = v[0]
                vect.append([x,y])
            transformation.append(vect)
        self.translations.append(transformation)
        ## Reflection across x-axis
        transformation = []
        for p in self.verts:
            vect = []
            for v in p:
                x = v[0]
                y = v[1] * -1
                vect.append([x,y])
            transformation.append(vect)
        self.translations.append(transformation)
        ## Reflection across y-axis
        transformation = []
        for p in self.verts:
            vect = []
            for v in p:
                x = v[0] * -1
                y = v[1]
                vect.append([x,y])
            transformation.append(vect)
        self.translations.append(transformation)
        ## Reflect x-coord and switch
        transformation = []
        for p in self.verts:
            vect = []
            for v in p:
                x = v[1]
                y = v[0] * -1
                vect.append([x,y])
            transformation.append(vect)
        self.translations.append(transformation)
        ## Reflect y-coord and switch
        transformation = []
        for p in self.verts:
            vect = []
            for v in p:
                x = v[1] * -1
                y = v[0]
                vect.append([x,y])
            transformation.append(vect)
        self.translations.append(transformation)
        ## Reflection across x-axis and y-axis
        transformation = []
        for p in self.verts:
            vect = []
            for v in p:
                x = v[0] * -1
                y = v[1] * -1
                vect.append([x,y])
            transformation.append(vect)
        self.translations.append(transformation)
        ## Reflection across x-axis and y-axis and switch
        transformation = []
        for p in self.verts:
            vect = []
            for v in p:
                x = v[1] * -1
                y = v[0] * -1
                vect.append([x,y])
            transformation.append(vect)
        self.translations.append(transformation)
        #### Fill each transformation list with it's permutations
        for idx, T in enumerate(self.translations):
            if len(self.verts[0]) != self.imbeddingDim:
                R = []
                for v in T:
                    vect = []
                    vect += v
                    for i in range(0, self.imbeddingDim - len(v) + 1):
                        if i != 0:
                            for k in range(i):
                                vect.insert(0,[0.,0.])
                        for j in range(len(vect), self.imbeddingDim, 1):
                            vect.append([0.,0.])
                        R.append(vect)
                        vect = []
                        vect += v
                self.translations[idx] = R
                R = []
        ### Vectorize each data point with it's 'picard number'
        R = []
        data = []
        for T in self.translations:
            for v in T:
                R.append((v, self.picardNum))
            data.append(R)
            R = []
        self.translations = data
