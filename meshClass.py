from GaussDict import Telles_function, gauss_legendre_table
from PhiDict import PHI_functions

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import gmsh
import sys

class MESH:
    def __init__(self, order=1, order_geom=1, gauss_points=4, a=0.15, b=0.15):
        self.NCORD = 0
        self.NDOMN = 0
        self.NNODE = np.empty(0, dtype=int)
        self.NELEM = np.empty(0, dtype=int)

        self.node = []
        self.cone = []
        self.k = np.array([])

        self.boundaryCond = []

        self.interface = []
        self.intDomain = np.empty((0, 2))
        self.intNumber = np.empty(0, dtype=int)


        self.domForces = np.empty(0, dtype=int)
        self.domForcesFunc = []
        self.domCellSize = []

        self.domSource = np.empty(0, dtype=int)
        self.domSourceLoc = np.empty((0, 2))
        self.domSourceMag = np.empty(0)

        self.sclu = np.NaN
        self.sclq = np.NaN

        self.internal_x0 = -1

        self.a = a
        self.b = b

        self.gauss_points = gauss_points
        self.order = order

        if self.order == 0:
            self.elem_nodes = np.array([0])
        if self.order >= 1:
            self.elem_nodes = np.linspace(-1 + self.a, 1 - self.b, self.order + 1)

        self.ξ = gauss_legendre_table[gauss_points]["xi"]
        self.w = gauss_legendre_table[gauss_points]["wi"]

        self.ξ_qs = gauss_legendre_table[64]["xi"]
        self.w_qs = gauss_legendre_table[64]["wi"]

        self.telles = Telles_function(self.ξ, self.w, 0)

        self.phi = PHI_functions[order]["PHI"]
        self.crt = PHI_functions[order]["CRT"](a, b) 

        self.phi_geom   = PHI_functions[order_geom]["PHI"]
        self.phi_geom_t = PHI_functions[order_geom]["dPHI"]

    def addNodes(self, cord):
        self.NCORD = len(cord)
        self.cord = cord

    def addDomain(self, cone, BC, k = 1):
        self.NDOMN += 1
        self.NELEM = np.append(self.NELEM, len(cone))
        self.NNODE = np.append(self.NNODE, len(cone) * (self.order + 1))

        self.cone.append(cone)
        self.node.append(self.phi_geom(self.elem_nodes).T@self.cord[cone])

        self.k = np.append(self.k, k)

        boundaryCond = np.ones((int(len(cone) * (self.order + 1)), 2)) * [1, 0]
        for ei, ni, ti, value in BC:
                boundaryCond[int((self.order + 1)*ei + ni)] = ti, value
        self.boundaryCond.append(boundaryCond)

        self.checkInterface()

    def addDomainForces(self, domId, forces, lc = 0.5):
        self.domForces = np.append(self.domForces, domId)
        self.domForcesFunc.append(forces)
        self.domCellSize.append(lc)

    def addDomainSource(self, domId, x, y, Q):

        self.domSource = np.append(self.domSource, domId)
        self.domSourceLoc = np.vstack([self.domSourceLoc, np.array([x, y])])
        self.domSourceMag = np.append(self.domSourceMag, Q)

    def checkInterface(self):

        coneF = self.cone[self.NDOMN - 1]
        for i in range(self.NDOMN - 1):
            coneI = self.cone[i]
            shared_rows = np.isin(coneI, coneF[:, [1, 0]]).all(axis=1)

            elemI = np.where(shared_rows)[0]

            shared_rows = np.isin(coneF[:, [1, 0]], coneI).all(axis=1)
            elemF = np.where(shared_rows)[0]

            if len(elemI) != 0:
                self.intNumber = np.append(self.intNumber, len(elemI)*(self.order + 1))
                nodeI = self.element2node(elemI)
                nodeF = self.element2node(elemF)[::-1]
                self.interface.append(np.vstack([nodeI, nodeF]))
                self.intDomain = np.vstack([self.intDomain, np.array([i, self.NDOMN - 1])])

    def element2node(self, arr):
        arr2 = []
        for x in arr:
            for i in range(self.order + 1):
                arr2.append(x*(self.order + 1) + i)
        return np.array(arr2)

    def solve(self):

        matrixPos = np.zeros(self.NDOMN + len(self.intNumber))
        for domID in range(self.NDOMN):
            bol = np.any(self.intDomain == domID, axis=1)
            matrixPos[domID] = self.NNODE[domID] - sum(self.intNumber[bol])

        matrixPos[self.NDOMN:] = self.intNumber*2
        matrixPos = np.hstack([0, np.cumsum(matrixPos)])
        matrixPos = matrixPos.astype(int)
        nnodes          = int(sum(self.NNODE))
        nnodesInterface = int(sum(self.intNumber)*2)

        A = np.empty(0)
        B = np.zeros((nnodes, nnodes))

        self.uSol = []
        self.qSol = []
        

        nnodes          = np.hstack([0, np.cumsum(self.NNODE).astype(int)])
        nnodesInterface = np.hstack([0, np.cumsum(self.intNumber).astype(int)])

        for domId in range(self.NDOMN):

            row_i = nnodes[domId]
            row_f = nnodes[domId + 1]

            AA, BB, DD = self.getMatrix(domId, self.boundaryCond[domId])

            notInterface = np.array([])

            index = np.argwhere(self.intDomain == domId)
            for i, j in index:
                notInterface = np.append(notInterface, self.interface[i][j])

                col_i = matrixPos[self.NDOMN + i]
                col_f = matrixPos[self.NDOMN + i + 1]
                

                B[row_i:row_f, col_i:col_f] = np.hstack([BB[:, self.interface[i][j]], 
                    (2*j - 1)*AA[:, self.interface[i][j]]])

            notInterface = np.setdiff1d(np.arange(self.NNODE[domId]), notInterface)

            col_i = matrixPos[domId]
            col_f = matrixPos[domId + 1]
            B[row_i:row_f, col_i:col_f] = BB[:, notInterface]
            A = np.append(A, AA[:, notInterface]@self.boundaryCond[domId][notInterface, 1] - DD)

        X = np.linalg.solve(B, A)

        for domId in range(self.NDOMN):

            u   = np.zeros(self.NNODE[domId])
            q   = self.boundaryCond[domId][:, 1]

            notInterface = np.array([])

            index = np.argwhere(self.intDomain == domId)
            for i, j in index:
                notInterface = np.append(notInterface, self.interface[i][j])

                col_i = matrixPos[self.NDOMN + i]
                col_f = matrixPos[self.NDOMN + i + 1]

                u[self.interface[i][j]] =           np.split(X[col_i:col_f], 2)[0]
                q[self.interface[i][j]] = (1 - 2*j)*np.split(X[col_i:col_f], 2)[1]

            col_i = matrixPos[domId]
            col_f = matrixPos[domId + 1]
            notInterface = np.setdiff1d(np.arange(self.NNODE[domId]), notInterface)

            u[notInterface] = X[col_i:col_f]

            for i in range(len(self.boundaryCond[domId])):
                if self.boundaryCond[domId][i, 0] == 0:
                    aux1 = q[i]
                    aux2 = u[i]
                    u[i] = aux1
                    q[i] = aux2

            self.uSol.append(u)
            self.qSol.append(q)

            if np.isnan(self.sclu) or np.abs(u).max() > self.sclu:
                self.sclu = np.abs(u).max()
            if np.isnan(self.sclq) or np.abs(q).max() > self.sclq:
                self.sclq = np.abs(q).max()

    def getMatrix(self, domainID, BC):
        NNODE = int(self.NNODE[domainID])
        NELEM = int(self.NELEM[domainID])
        node  = self.node[domainID]
        cone  = self.cone[domainID]
        k     = self.k[domainID]

        H = np.zeros((NNODE, NNODE))
        G = np.zeros((NNODE, NNODE))
        A = np.zeros((NNODE, NNODE))
        B = np.zeros((NNODE, NNODE))
        D = np.zeros(NNODE)

        ξ , w      = self.ξ, self.w
        PHI        = self.crt @ self.phi(ξ)
        PHI_geom   = self.phi_geom(ξ)
        PHI_geom_t = self.phi_geom_t(ξ)

        domFlag = 0
        if np.isin(domainID, self.domForces):
            position = np.where(self.domForces == domainID)[0][0]
            forces = self.domForcesFunc[position]
            lc     = self.domCellSize[position]
            element_areas, element_centroids = self.internalCell(domainID, lc = lc)
            domFlag = 1

        souFlag = 0
        if np.isin(domainID, self.domSource):
            position = np.where(self.domSource == domainID)[0]
            if len(position) == 1:
                position = position[0]

            sourceLoc = self.domSourceLoc[position]
            sourceMag = self.domSourceMag[position]
            souFlag = 1

        for ni in range(self.order + 1):
                      
            ξt, wt      = self.telles(self.elem_nodes[ni])
            PHIT        = self.crt @ self.phi(ξt)
            PHIT_geom   = self.phi_geom(ξt)
            PHIT_geom_t = self.phi_geom_t(ξt)
            
            for ei in range(NELEM):
                ii = ei * (self.order + 1) + ni
                x0, y0 = node[ei, ni]

                if domFlag == 1:
                    
                    xx, yy = element_centroids.T
                    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
                    val =  -(1 / 2 / np.pi * np.log(r))*forces(xx, yy)*element_areas
                    D[ii] += sum(val)

                if souFlag == 1:
                    
                    xx, yy = sourceLoc.T
                    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

                    val =  -(1 / 2 / np.pi * np.log(r))*sourceMag
                    if isinstance(val, (list, np.ndarray)):

                        D[ii] += sum(val)
                    else:
                        D[ii] += val

                for ej in range(NELEM):
                    
                    cord   = self.cord[cone[ej]]                  
                    jj     = ej * (self.order + 1) + np.arange(self.order + 1)

                    # Elemento COM singularidade
                    if ei == ej:
                        tx, ty = cord.T@PHIT_geom_t
                        J      = np.sqrt(tx**2 + ty**2)
                        nx, ny = (ty, -tx)/J
                        xx, yy = cord.T @ PHIT_geom
                        
                        r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
                        drdn = (xx - x0) * nx + (yy - y0) * ny
                        drdn /= r
                        
                        G[ii, jj] +=   -(1 / 2 / np.pi * np.log(r) * J) @ (wt * PHIT).T
                        H[ii, jj] += -k*(1 / 2 / np.pi / r * drdn  * J) @ (wt * PHIT).T
                        # G[ii, jj] += self.le[e_j]/(2*np.pi)*(np.log(2/self.le[e_j]) + 1)

                    # Elemento SEM singularidade
                    if ei != ej:
                        tx, ty = cord.T@PHI_geom_t
                        J      = np.sqrt(tx**2 + ty**2)
                        nx, ny = (ty, -tx)/J
                        xx, yy = cord.T @ PHI_geom
                        
                        r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
                        drdn = (xx - x0) * nx + (yy - y0) * ny
                        drdn /= r

                        G[ii, jj] +=   -(1 / 2 / np.pi * np.log(r) * J) @ (w * PHI).T
                        H[ii, jj] += -k*(1 / 2 / np.pi / r * drdn  * J) @ (w * PHI).T

                H[ii, ii] = -sum(H[ii, :])

        for i in range(len(BC)):
            if BC[i, 0] == 0:
                A[:, i] = -H[:, i]
                B[:, i] = -G[:, i]
            if BC[i, 0] == 1:
                A[:, i] = G[:, i]
                B[:, i] = H[:, i]

        return A, B, D


    def internalCell(self, domId, lc = 0.5, view = False):
        # Initialize gmsh
        gmsh.initialize()

        # Define the 2D points using a NumPy array (example for a square)
        points = self.cord[self.cone[domId][:, 0]]

        # Create points in GMSH from NumPy array
        point_ids = []
        for x, y in points:
            point_ids.append(gmsh.model.geo.add_point(x, y, 0, lc))  # z = 0 for 2D

        # Define the lines (edges) in order to form a closed loop
        lines = []
        for i in range(len(points)):
            lines.append((i, (i + 1) % len(points)))  # Wrap around to form a closed loop

        line_ids = []
        for p1, p2 in lines:
            line_ids.append(gmsh.model.geo.add_line(point_ids[p1], point_ids[p2]))

        # Create a surface enclosed by the lines
        curve_loop = gmsh.model.geo.add_curve_loop(line_ids)
        surface = gmsh.model.geo.add_plane_surface([curve_loop])

        # Synchronize the model with GMSH data structures
        gmsh.model.geo.synchronize()

        # Generate the mesh
        gmsh.model.mesh.generate(2)  # Specify 2D mesh generation

        # Get the element data
        node_coords = {}  # Store coordinates of nodes
        element_centroids = []  # Store centroids of elements
        element_areas = []  # Store areas of elements

        # Get nodes
        node_tags, node_coords_flat, _ = gmsh.model.mesh.get_nodes()
        node_coords = {tag: node_coords_flat[i:i + 3] for i, tag in zip(range(0, len(node_coords_flat), 3), node_tags)}

        # Get elements
        element_types, element_tags, node_tags_by_element = gmsh.model.mesh.get_elements()

        # Loop through elements to compute centroids and areas
        for etype, etags, elem_nodes in zip(element_types, element_tags, node_tags_by_element):
            if etype == 2:  # 2 means triangular element
                for elem, nodes in zip(etags, np.array_split(elem_nodes, len(etags))):
                    # Get the coordinates of the nodes for the triangle
                    coords = np.array([node_coords[n] for n in nodes])
                    x_coords, y_coords = coords[:, 0], coords[:, 1]
                    # Compute the area using the determinant formula
                    area = 0.5 * abs(
                        x_coords[0] * (y_coords[1] - y_coords[2]) +
                        x_coords[1] * (y_coords[2] - y_coords[0]) +
                        x_coords[2] * (y_coords[0] - y_coords[1])
                    )
                    element_areas.append(area)

                    # Compute the centroid as the average of the nodes' coordinates
                    centroid_x = np.mean(x_coords)
                    centroid_y = np.mean(y_coords)
                    element_centroids.append((centroid_x, centroid_y))
        element_areas     = np.array(element_areas)
        element_centroids = np.array(element_centroids)

        # Optional: Show the GUI for visualization (if not running with 'close')

        if 'close' not in sys.argv and view == True:
            gmsh.fltk.run()

        # Finalize GMSH
        gmsh.finalize()

        return element_areas, element_centroids

    def internal(self, domainID, x):

        NNODE = int(self.NNODE[domainID])
        NELEM = int(self.NELEM[domainID])
        node  = self.node[domainID]
        cone  = self.cone[domainID]
        k     = self.k[domainID]

        Huu = np.zeros((len(x), NNODE))
        Guu = np.zeros((len(x), NNODE))
        Duu = np.zeros(len(x))

        Hqx = np.zeros((len(x), NNODE))
        Gqx = np.zeros((len(x), NNODE))
        Dqx = np.zeros(len(x))

        Hqy = np.zeros((len(x), NNODE))
        Gqy = np.zeros((len(x), NNODE))
        Dqy = np.zeros(len(x))

        
        ξ , w      = self.ξ, self.w
        PHI        = self.crt @ self.phi(ξ)
        PHI_geom   = self.phi_geom(ξ)
        PHI_geom_t = self.phi_geom_t(ξ)

        domFlag = 0
        if np.isin(domainID, self.domForces):
            position = np.where(self.domForces == domainID)[0][0]
            forces = self.domForcesFunc[position]
            lc     = self.domCellSize[position]
            element_areas, element_centroids = self.internalCell(domainID, lc = lc)
            domFlag = 1

        souFlag = 0
        if np.isin(domainID, self.domSource):
            position = np.where(self.domSource == domainID)[0]
            if len(position) == 1:
                position = position[0]

            sourceLoc = self.domSourceLoc[position]
            sourceMag = self.domSourceMag[position]
            souFlag = 1
        
        for ii, [x0, y0] in enumerate(x):

            if domFlag == 1:
                
                xx, yy = element_centroids.T
                r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
                drdx = (xx - x0)/r
                drdy = (yy - y0)/r

                val_uu = -(1 / 2 / np.pi * np.log(r))*forces(xx, yy)*element_areas
                val_qx =  (1 / 2 / np.pi / r * drdx)*forces(xx, yy)*element_areas
                val_qy =  (1 / 2 / np.pi / r * drdy)*forces(xx, yy)*element_areas

                Duu[ii] += sum(val_uu)
                Dqx[ii] += sum(val_qx)
                Dqy[ii] += sum(val_qy)

            if souFlag == 1:
                
                xx, yy = sourceLoc.T
                r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
                drdx = (xx - x0)/r
                drdy = (yy - y0)/r

                val_uu = -(1 / 2 / np.pi * np.log(r))*sourceMag
                val_qx =  (1 / 2 / np.pi / r * drdx)*sourceMag
                val_qy =  (1 / 2 / np.pi / r * drdy)*sourceMag

                if isinstance(val_uu, (list, np.ndarray)):
                    Duu[ii] += sum(val_uu)
                    Dqx[ii] += sum(val_qx)
                    Dqy[ii] += sum(val_qy)
                else:
                    Duu[ii] += val_uu
                    Dqx[ii] += val_qx
                    Dqy[ii] += val_qy

            for ej in range(NELEM):

                cord   = self.cord[cone[ej]]
                jj     = ej * (self.order + 1) + np.arange(self.order + 1)

                tx, ty = cord.T@PHI_geom_t
                J      = np.sqrt(tx**2 + ty**2)
                nx, ny = (ty, -tx)/J
                xx, yy = cord.T @ PHI_geom

                r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

                drdx = (xx - x0)/r
                drdy = (yy - y0)/r

                drdn  = drdx * nx + drdy * ny

                aux_x = (2*drdx**2 - 1)*nx + (2*drdx*drdy)*ny
                aux_y = (2*drdy**2 - 1)*ny + (2*drdx*drdy)*nx
                
                Guu[ii, jj] +=    -(1 / 2 / np.pi * np.log(r) * J) @ (w * PHI).T
                Huu[ii, jj] +=   -k*(1 / 2 / np.pi / r * drdn  * J) @ (w * PHI).T
                Hqx[ii, jj] +=   -k*(1 / 2 / np.pi / r**2 * aux_x * J) @ (w * PHI).T
                Hqy[ii, jj] +=   -k*(1 / 2 / np.pi / r**2 * aux_y * J) @ (w * PHI).T
                Gqx[ii, jj] +=   (1 / 2 / np.pi / r * drdx * J) @ (w * PHI).T
                Gqy[ii, jj] +=   (1 / 2 / np.pi / r * drdy * J) @ (w * PHI).T

       
        self.internal_x0 =  x
        self.internal_u  =  Guu@self.qSol[domainID] - Huu@self.uSol[domainID] - Duu
        self.internal_qx =  Gqx@self.qSol[domainID] - Hqx@self.uSol[domainID] - Dqx
        self.internal_qy =  Gqy@self.qSol[domainID] - Hqy@self.uSol[domainID] - Dqy

        #return self.internal_u, self.internal_qx, self.internal_qy

    def plot(self, sol=False, scale=1, internal = False, offset = 0):
        ξ = np.linspace(-1, 1, 101)

        PHI        = self.crt @ self.phi(ξ)
        PHI_geom   = self.phi_geom(ξ)
        PHI_geom_t = self.phi_geom_t(ξ)

        if self.sclu == 0:
            self.sclu = 1
        if self.sclq == 0:
            self.sclq = 1
        
        fig, ax = plt.subplots(figsize=(10, 6))

        for p in self.cord:
            plt.scatter(*p, color="r", marker="+")


        for domId in range(self.NDOMN):
            for ei in range(self.NELEM[domId]):      
                cord = self.cord[self.cone[domId][ei]]
                ax.scatter(*self.node[domId][ei].T, color="k", s=5)
                
                xx, yy = cord.T @ PHI_geom
                ax.plot(xx, yy, color="blue", zorder=-1)


        if sol == True:
            ls = '--'
            if offset != 0 :
                ls = '-'
                for p in self.cord:
                    paux = p + np.array([offset, 0])
                    plt.scatter(*paux, color="r", marker="+")

                for domId in range(self.NDOMN):
                    for i, cone in enumerate(self.cone[domId]):

                        cord = self.cord[cone]
                        xx, yy = cord.T @ PHI_geom
                        ax.plot(xx + offset, yy, color="blue", zorder=-1)


                        #paux = self.cord[cone] +  np.tile(np.array([offset, 0]), (2, 1))
                        #ax.plot(*paux.T, color="blue", zorder=-1)
                        
                        paux = self.node[domId][i]    +  np.tile(np.array([offset, 0]), (len(self.node[domId][i]), 1))
                        ax.scatter(*paux.T, color="k", s=5)

            for domId in range(self.NDOMN):
                for ei in range(self.NELEM[domId]):               
                    cord = self.cord[self.cone[domId][ei]]

                    ii = ei * (self.order + 1) + np.arange(self.order + 1)

                    uSol = self.uSol[domId][ii]
                    qSol = self.qSol[domId][ii]
                    
                    tx, ty = cord.T@PHI_geom_t
                    J      = np.sqrt(tx**2 + ty**2)
                    nx, ny = (ty, -tx)/J
                    xx, yy = cord.T @ PHI_geom
                    
                    ii = ei * (self.order + 1) + np.arange(self.order + 1)

                    uSol = uSol @ PHI / self.sclu * scale
                    qSol = qSol @ PHI / self.sclq * scale

                    Uλx, Uλy = xx + uSol * nx, yy + uSol * ny
                    Qλx, Qλy = xx + qSol * nx, yy + qSol * ny

                    plt.plot([xx[ 0], Uλx[ 0]], [yy[ 0], Uλy[ 0]], color="k")
                    plt.plot([xx[-1], Uλx[-1]], [yy[-1], Uλy[-1]], color="k")
                    plt.plot(Uλx, Uλy                            , color="k")


                    plt.plot([xx[ 0]+ offset, Qλx[ 0]+ offset], [yy[ 0], Qλy[ 0]], color="orange", ls = ls)
                    plt.plot([xx[-1]+ offset, Qλx[-1]+ offset], [yy[-1], Qλy[-1]], color="orange", ls = ls)
                    plt.plot(Qλx + offset, Qλy                                   , color="orange", ls = ls)

            if internal == True:
                qmax = np.sqrt(np.amax(self.internal_qx**2 + self.internal_qy**2))

                if qmax == 0:
                    qmax = 1
                

                for i, x0 in enumerate(self.internal_x0):

                    u  = self.internal_u[i] 
                    qx = self.internal_qx[i]
                    qy = self.internal_qy[i]

                    plt.text(*x0, "{:.2f}".format(u), color='green', fontsize=8, ha='left', va='bottom')
                    plt.text(*x0 + [offset, 0], "({:.2f}, {:.2f})".format(qx, qy), color='green', fontsize=8, ha='left', va='bottom')

                    qx = qx / qmax * scale
                    qy = qy / qmax * scale

                    plt.scatter(*x0, s = 5, color = 'g')
                    
                    plt.arrow(*x0 + [offset, 0], qx, qy, head_width=0.05, head_length=0.1, color = 'g')

        plt.axis("equal")
        plt.show()