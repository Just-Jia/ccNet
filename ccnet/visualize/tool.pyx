# from math import sqrt, log, pow
from libc.math cimport sqrt, log, pow

cdef class Node:
    cdef public double mass, size, old_dx, old_dy, dx, dy, x, y
    def __init__(self):
        self.mass = 0.0
        self.size = 0.0
        self.old_dx = 0.0
        self.old_dy = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.x = 0.0
        self.y = 0.0
    def show(self):
        print('mass   = {}'.format(self.mass))
        print('size   = {}'.format(self.size))
        print('old_dx = {}'.format(self.old_dx))
        print('old_dy = {}'.format(self.old_dy))
        print('dx     = {}'.format(self.dx))
        print('dy     = {}'.format(self.dy))
        print('x      = {}'.format(self.x))
        print('y      = {}'.format(self.y))

cdef class Edge:
    cdef public int node1, node2
    cdef public double weight
    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0

# For Barnes Hut Optimization
cdef class Region:
    cdef:
        public double mass, massCenterX, massCenterY, size
        public list nodes, subregions
    def __init__(self, nodes):
        self.mass = 0.0
        self.massCenterX = 0.0
        self.massCenterY = 0.0
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        self.updateMassAndGeometry()


    cdef void updateMassAndGeometry(self):
        cdef:
            double massSumX, massSumY, distance
            Node n

        if len(self.nodes) > 1:
            self.mass = 0
            massSumX = 0
            massSumY = 0
            for n in self.nodes:
                self.mass += n.mass
                massSumX += n.x * n.mass
                massSumY += n.y * n.mass
            self.massCenterX = massSumX / self.mass
            self.massCenterY = massSumY / self.mass

            self.size = 0.0
            for n in self.nodes:
                distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
                self.size = max(self.size, 2 * distance)


    cpdef void buildSubRegions(self):
        cdef:
            list topleftNodes, bottomleftNodes, toprightNodes, bottomrightNodes
            Node n
            Region subregion

        if len(self.nodes) > 1:
            topleftNodes = []
            bottomleftNodes = []
            toprightNodes = []
            bottomrightNodes = []
            # Optimization: The distribution of self.nodes into 
            # subregions now requires only one for loop. Removed 
            # topNodes and bottomNodes arrays: memory space saving.
            for n in self.nodes:
                if n.x < self.massCenterX:
                    if n.y < self.massCenterY:
                        bottomleftNodes.append(n)
                    else:
                        topleftNodes.append(n)
                else:
                    if n.y < self.massCenterY:
                        bottomrightNodes.append(n)
                    else:
                        toprightNodes.append(n)      

            if len(topleftNodes) > 0:
                if len(topleftNodes) < len(self.nodes):
                    subregion = Region(topleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in topleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomleftNodes) > 0:
                if len(bottomleftNodes) < len(self.nodes):
                    subregion = Region(bottomleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(toprightNodes) > 0:
                if len(toprightNodes) < len(self.nodes):
                    subregion = Region(toprightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in toprightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomrightNodes) > 0:
                if len(bottomrightNodes) < len(self.nodes):
                    subregion = Region(bottomrightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomrightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            for subregion in self.subregions:
                subregion.buildSubRegions()


    cdef void applyForce(self, Node n, double theta, double coefficient=0, bint adjust_sizes=False):
        cdef:
            double distance
            Region subregion

        if len(self.nodes) < 2:
            linRepulsion(n, self.nodes[0], coefficient, adjust_sizes)
        else:
            distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
            if distance * theta > self.size:
                linRepulsion_region(n, self, coefficient, adjust_sizes)
            else:
                for subregion in self.subregions:
                    subregion.applyForce(n, theta, coefficient, adjust_sizes)


    cpdef void applyForceOnNodes(self, list nodes, double theta, double coefficient=0, bint adjust_sizes=False):
        cdef Node n

        for n in nodes:
            self.applyForce(n, theta, coefficient)


cpdef void linRepulsion(Node n1, Node n2, double coefficient=0, bint adjust_sizes=False):
    cdef double xDist, yDist, distance, distance2, factor
    
    xDist = n1.x - n2.x
    yDist = n1.y - n2.y

    if adjust_sizes:
        distance = sqrt(xDist * xDist + yDist * yDist) - n1.size - n2.size
        if distance > 0:
            factor = coefficient * n1.mass * n2.mass / ( distance * distance )
        elif distance < 0:
            factor = 100 * coefficient * n1.mass * n2.mass
        else:
            factor = 0
    else:

        distance2 = xDist * xDist + yDist * yDist  # Distance squared

        if distance2 > 0:
            factor = coefficient * n1.mass * n2.mass / distance2
        else:
            factor = 0
            
    n1.dx += xDist * factor
    n1.dy += yDist * factor
    n2.dx -= xDist * factor
    n2.dy -= yDist * factor

    # original codes
    # xDist = n1.x - n2.x
    # yDist = n1.y - n2.y
    # distance2 = xDist * xDist + yDist * yDist  # Distance squared

    # if distance2 > 0:
    #     factor = coefficient * n1.mass * n2.mass / distance2
    #     n1.dx += xDist * factor
    #     n1.dy += yDist * factor
    #     n2.dx -= xDist * factor
    #     n2.dy -= yDist * factor




cdef void linRepulsion_region(Node n, Region r, double coefficient=0, bint adjust_sizes=False):
    cdef double xDist, yDist, distance, distance2, factor

    xDist = n.x - r.massCenterX
    yDist = n.y - r.massCenterY

    # refer to https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceFactory.java#L210
    if adjust_sizes:
        # 原先没有减两个size，但我觉得应该减去
        # distance = sqrt(xDist * xDist + yDist * yDist)
        distance = sqrt(xDist * xDist + yDist * yDist) - n.size - r.size
        if distance > 0:
            factor = coefficient * n.mass * r.mass / distance / distance
        elif distance < 0:
            factor = -coefficient * n.mass * r.mass / distance
    else:
        distance2 = xDist * xDist + yDist * yDist  # Distance squared
        if distance2 > 0:
            factor = coefficient * n.mass * r.mass / distance2
    
    n.dx += xDist * factor
    n.dy += yDist * factor


cdef void linGravity(Node n, double g):
    cdef double xDist, yDist, distance, factor

    xDist = n.x
    yDist = n.y
    distance = sqrt(xDist * xDist + yDist * yDist)

    if distance > 0:
        factor = n.mass * g / distance
        n.dx -= xDist * factor
        n.dy -= yDist * factor


cdef void strongGravity(Node n, double g, double coefficient=0):
    cdef double xDist, yDist, factor

    xDist = n.x
    yDist = n.y

    factor = coefficient * n.mass * g
    n.dx -= xDist * factor
    n.dy -= yDist * factor

    # if xDist != 0 and yDist != 0:
    #     factor = coefficient * n.mass * g
    #     n.dx -= xDist * factor
    #     n.dy -= yDist * factor


cpdef void linAttraction(Node n1, Node n2, double e, bint distributedAttraction, double coefficient=0):
    cdef double xDist, yDist, factor

    xDist = n1.x - n2.x
    yDist = n1.y - n2.y
    if not distributedAttraction:
        factor = -coefficient * e
    else:
        factor = -coefficient * e / n1.mass
    n1.dx += xDist * factor
    n1.dy += yDist * factor
    n2.dx -= xDist * factor
    n2.dy -= yDist * factor


cpdef void apply_repulsion(list nodes, double coefficient, bint adjust_sizes=False):
    cdef:
        int i, j
        Node n1, n2

    i = 0
    for n1 in nodes:
        j = i
        for n2 in nodes:
            if j == 0:
                break
            linRepulsion(n1, n2, coefficient, adjust_sizes)
            j -= 1
        i += 1


cpdef void apply_gravity(list nodes, double gravity, double scalingRatio, bint useStrongGravity=False):
    cdef Node n

    if useStrongGravity:
        for n in nodes:
            strongGravity(n, gravity, scalingRatio)
    else:
        for n in nodes:
            linGravity(n, gravity)

    # if not useStrongGravity:
    #     for n in nodes:
    #         linGravity(n, gravity)
    # else:
    #     for n in nodes:
    #         strongGravity(n, gravity, scalingRatio)

cpdef void apply_attraction(list nodes, list edges, double compensation, double weight_influence,
    bint log_attraction, bint dist_attraction, bint adjust_sizes):
    # Optimization, since usually edgeWeightInfluence is 0 or 1, and pow is slow
    cdef Edge edge

    if weight_influence == 0:
        for edge in edges:
            attraction_force(nodes[edge.node1], nodes[edge.node2], compensation, 1,
                log_attraction, dist_attraction, adjust_sizes)
    elif weight_influence == 1:
        for edge in edges:
            attraction_force(nodes[edge.node1], nodes[edge.node2], compensation, edge.weight,
                log_attraction, dist_attraction, adjust_sizes)
    else:
        for edge in edges:
            attraction_force(nodes[edge.node1], nodes[edge.node2], compensation, pow(edge.weight, weight_influence),
                log_attraction, dist_attraction, adjust_sizes)

cdef void attraction_force(Node n1, Node n2, double compensation, double weight,
    bint log_attraction, bint dist_attraction, bint adjust_sizes):
    cdef double xDist, yDist, distance, factor

    xDist = n1.x - n2.x
    yDist = n1.y - n2.y

    distance = sqrt(xDist * xDist + yDist * yDist)

    if adjust_sizes:
        distance = distance - n1.size - n2.size

    if distance > 0:
        if log_attraction:
            if dist_attraction:
                factor = -compensation * weight * log(1+distance) / distance / n1.mass
            else:
                factor = -compensation * weight * log(1+distance) / distance
        else:
            if dist_attraction:
                factor = -compensation * weight  / n1.mass
            else:
                factor = -compensation * weight
    else:
        factor = 0
   
    n1.dx += xDist * factor
    n1.dy += yDist * factor
    n2.dx -= xDist * factor
    n2.dy -= yDist * factor


    # if edgeWeightInfluence == 0:
    #     for edge in edges:
    #         linAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
    # elif edgeWeightInfluence == 1:
    #     for edge in edges:
    #         linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
    # else:
    #     for edge in edges:
    #         linAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
    #                       distributedAttraction, coefficient)

# cpdef void apply_attraction(list nodes, list edges, bint distributedAttraction, double coefficient, double edgeWeightInfluence):
#     # Optimization, since usually edgeWeightInfluence is 0 or 1, and pow is slow
#     cdef Edge edge

#     if edgeWeightInfluence == 0:
#         for edge in edges:
#             linAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
#     elif edgeWeightInfluence == 1:
#         for edge in edges:
#             linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
#     else:
#         for edge in edges:
#             linAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
#                           distributedAttraction, coefficient)


cpdef dict adjustSpeedAndApplyForces(list nodes, double speed, double speedEfficiency, double jitterTolerance, bint adjust_sizes):
    # Auto adjust speed.
    cdef:
        double swinging, totalSwinging, totalEffectiveTraction, estimatedOptimalJitterTolerance, \
                minJT, maxJT, jt, minSpeedEfficiency, targetSpeed, maxRise, factor
        Node n
        dict values

    totalSwinging = 0.0  # How much irregular movement
    totalEffectiveTraction = 0.0  # How much useful movement
    for n in nodes:
        swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        totalSwinging += n.mass * swinging
        totalEffectiveTraction += .5 * n.mass * sqrt(
            (n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))

    # Optimize jitter tolerance.  The 'right' jitter tolerance for
    # this network. Bigger networks need more tolerance. Denser
    # networks need less tolerance. Totally empiric.
    estimatedOptimalJitterTolerance = .05 * sqrt(len(nodes))
    minJT = sqrt(estimatedOptimalJitterTolerance)
    maxJT = 10
    jt = jitterTolerance * max(minJT,
                               min(maxJT, estimatedOptimalJitterTolerance * totalEffectiveTraction / (
                                   len(nodes) * len(nodes))))

    minSpeedEfficiency = 0.05

    # Protective against erratic behavior
    if totalEffectiveTraction and totalSwinging / totalEffectiveTraction > 2.0:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .5
        jt = max(jt, jitterTolerance)

    if totalSwinging == 0:
        targetSpeed = float('inf')
    else:
        targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

    if totalSwinging > jt * totalEffectiveTraction:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .7
    elif speed < 1000:
        speedEfficiency *= 1.3

    # But the speed shoudn't rise too much too quickly, since it would
    # make the convergence drop dramatically.
    maxRise = .5
    speed = speed + min(targetSpeed - speed, maxRise * speed)

    # Apply forces.
    #
    # Need to add a case if adjustSizes ("prevent overlap") is
    # implemented.
    if adjust_sizes:
        #If nodes overlap prevention is active, it's not possible to trust the swinging measure
        for n in nodes:
            swinging = n.mass * sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
            factor = (.1 * speed) / (1.0 + sqrt(speed * swinging))

            df = sqrt(n.dx * n.dx + n.dy * n.dy) + 1e-6 # avoid df equal 0
            factor = min(factor * df, 10) / df

            n.x = n.x + (n.dx * factor)
            n.y = n.y + (n.dy * factor)

    else:
        for n in nodes:
            swinging = n.mass * sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
            factor = speed / (1.0 + sqrt(speed * swinging))
            n.x = n.x + (n.dx * factor)
            n.y = n.y + (n.dy * factor)

    values = {}
    values['speed'] = speed
    values['speedEfficiency'] = speedEfficiency

    return values