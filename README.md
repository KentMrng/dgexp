# dgexp

**This script extract from "cmt/dge.py" created by Chad Vernon.**

https://github.com/chadmv/cmt

--------------------------------------------------------------------------------


Allow the creation of node networks through equation strings.

Dependency Graph EXPression (dgexp) is a convenience API used to simplify the creation of Maya DG node networks.
Rather than scripting out many createNode, get/setAttr, connectAttr commands, you can specify a string equation.

No complied plug-ins are used. All created nodes are vanilla Maya nodes.
Each created node has notes added to it to describe its place in the equation.

--------------------------------------------------------------------------------
## Example Usage
```python
from dgexp import dgexp

# Create a simple mathematical graph
loc = cmds.spaceLocator()[0]
result = dgexp("(x+3)*(2+x)", x=f"{loc}.tx")
cmds.connectAttr(result, f"{loc}.ty")

# Use assignment operator to auto-connect
loc = cmds.spaceLocator()[0]
dgexp("y=x^2", x=f"{loc}.tx", y=f"{loc}.ty")

# More complex example with ternary operator and functions
soft_ik_percentage = dgexp(
    "x > (1.0 - softIk) ? (1.0 - softIk) + softIk * (1.0 - exp(-(x - (1.0 - softIk)) / softIk)) : x",
    x=f"{stretch_scale_mdn}.outputX",
    softIk=f"{ik_control}.softIk
)
# Put the created nodes in a container
soft_ik_percentage = dgexp(
    "x > (1.0 - softIk) ? (1.0 - softIk) + softIk * (1.0 - exp(-(x - (1.0 - softIk)) / softIk)) : x",
    container="softik_container",
    x=f"{stretch_scale_mdn}.outputX",
    softIk=f"{ik_control}.softIk
)
```

-------------------------------------------------------------------------------
## Supported Syntax

### Operations:
    +  # addition
    -  # subtraction
    *  # multiplication
    /  # division
    ^  # power
    ?: # ternary

### Functions:
    abs(x)
    exp(x)
    clamp(x, min, max)
    lerp(a, b, t)
    min(x, y)
    max(x, y)
    sqrt(x)
    cos(x)
    sin(x)
    tan(x)
    acos(x)
    asin(x)
    atan(x)
    distance(node1, node2)
    hypot(x, y)
    not(x)
    and(x, y)
    nand(x, y)
    or(x, y)
    nor(x, y)
    xor(x, y)
    xnor(x, y)

### Constants:
    PI
    E

-------------------------------------------------------------------------------
## Before & After

### Before:

```python
# (1.0 - softik)
one_minus = cmds.createNode("plusMinusAverage", name=f"{name}_one_minus_softik")
cmds.setAttr(f"{one_minus}.operation", 2)
cmds.setAttr(f"{one_minus}.input1D[0]", 1)
cmds.connectAttr(softik, f"{one_minus}.input1D[1])

# x - (1.0 - softik)
x_minus = cmds.createNode("plusMinusAverage", name=f"{name}_x_minus_one_minus_softik")
cmds.setAttr(f"{x_minus}.operation", 2)
cmds.connectAttr(percent_rest_distance, f"{x_minus}.input1D[0]")
cmds.connectAttr(f"{one_minus}.output1D", f"{x_minus}.input1D[1]")

# -(x - (1.0 - softik))
negate = cmds.createNode("multDoubleLinear", name=f"{name}_softik_negate")
cmds.setAttr(f"{negate}.input1", -1)
cmds.connectAttr(f"{x_minus}.output1D", f"{negate}.input2")

# -(x - (1.0 - softik)) / softik
divide = cmds.createNode("multiplyDivide", name=f"{name}_softik_divide")
cmds.setAttr(f"{divide}.operation", 2)  # divide
cmds.connectAttr(f"{negate}.output", f"{divide}.input1X")
cmds.connectAttr(softik, f"{divide}.input2X")

# exp(-(x - (1.0 - softIk)) / softIk)
exp = cmds.createNode("multiplyDivide", name=f"{name}_softik_exp")
cmds.setAttr(f"{exp}.operation", 3)  # pow
cmds.setAttr(f"{exp}.input1X", 2.71828)
cmds.connectAttr(f"{divide}.outputX", f"{exp}.input2X")

# 1.0 - exp(-(x - (1.0 - softIk)) / softIk)
one_minus_exp = cmds.createNode("plusMinusAverage", name=f"{name}_one_minus_exp")
cmds.setAttr(f"{one_minus_exp}.operation", 2)
cmds.setAttr(f"{one_minus_exp}.input1D[0]", 1)
cmds.connectAttr(f"{exp}.outputX", f"{one_minus_exp}.input1D[1]")

# softik * (1.0 - exp(-(x - (1.0 - softIk)) / softIk))
mdl = cmds.createNode("multDoubleLinear", name=f"{name}_softik_mdl")
cmds.connectAttr(softik, f"{mdl}.input1")
cmds.connectAttr(f"{one_minus_exp}.output1D", f"{mdl}.input2")

# (1.0 - softik) + softik * (1.0 - exp(-(x - (1.0 - softIk)) / softIk))
adl = cmds.createNode("addDoubleLinear", name=f"{name}_softik_adl")
cmds.connectAttr(f"{one_minus}.output1D", f"{adl}.input1")
cmds.connectAttr(f"{mdl}.output", f"{adl}.input2")
# Now output of adl is the % of the rest distance the ik handle should be from the start joint

# Only adjust the ik handle if it is less than the soft percentage threshold
cnd = cmds.createNode("condition", name=f"{name}_current_length_greater_than_soft_length")
cmds.setAttr(f"{cnd}.operation", 2)  # greater than
cmds.connectAttr(percent_rest_distance, f"{cnd}.firstTerm")
cmds.connectAttr(f"{one_minus}.output1D", f"{cnd}.secondTerm")
cmds.connectAttr(f"{adl}.output", f"{cnd}.colorIfTrueR")
cmds.connectAttr(percent_rest_distance, f"{cnd}.colorIfFalseR")

softik_percentage = "{}.outColorR".format(cnd)
```

### After:

```python
soft_ik_percentage = dge(
    "x > (1.0 - softIk) ? (1.0 - softIk) + softIk * (1.0 - exp(-(x - (1.0 - softIk)) / softIk)) : x",
    container=f"{name}_softik",
    x=percent_rest_distance,
    softIk=softik,
)
```
