"""Allow the creation of node networks through equation strings.

Dependency Graph EXPression (dgexp) is a convenience API used to simplify the creation
of Maya DG node networks. Rather than scripting out many createNode, get/setAttr,
connectAttr commands, you can specify a string equation.

No complied plug-ins are used. All created nodes are vanilla Maya nodes.
Each created node has notes added to it to describe its place in the equation.

#--------------------------------------------------------------------------------
# Example Usage
#--------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
# Supported Syntax
#-------------------------------------------------------------------------------
Operations:
    +  # addition
    -  # subtraction
    *  # multiplication
    /  # division
    ^  # power
    ?: # ternary

Functions:
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

Constants:
    PI
    E

#-------------------------------------------------------------------------------
# Before & After
#-------------------------------------------------------------------------------

## Before:

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


## After:

soft_ik_percentage = dge(
    "x > (1.0 - softIk) ? (1.0 - softIk) + softIk * (1.0 - exp(-(x - (1.0 - softIk)) / softIk)) : x",
    container=f"{name}_softik",
    x=percent_rest_distance,
    softIk=softik,
)


)

"""
import math

from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    CaselessKeyword,
    Suppress,
    delimitedList,
    oneOf,
    Optional,
    FollowedBy,
)

from maya import cmds


def dgexp(expression, container=None, **kwargs):
    parser = DGParser()
    return parser.eval(expression, container=container, **kwargs)


class DGParser(object):
    def __init__(self):
        self.kwargs = {}
        self.expression_stack = []
        self.assignment_stack = []
        self.expression_string = None
        self.results = None
        self.container = None

        # Look up to optimize redundant nodes
        self.created_nodes = {}

        self.operations = {
            "+": self.add,
            "-": self.subtract,
            "/": self.divide,
            "^": self.pow,
        }

        self.functions = {
            "abs": self.abs,
            "exp": self.exp,
            "clamp": self.clamp,
            "min": self.min,
            "max": self.max,
            "sqrt": self.sqrt,
            "cos": self.cos,
            "sin": self.sin,
            "tan": self.tan,
            "acos": self.acos,
            "asin": self.asin,
            "atan": self.atan,
            "distance": self.distance,
        }

        self.conditionals =["==", "!=", ">", ">=", "<", "<="]

        # Use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with "e" or "pi" (such as "exp");
        # Keyword and CaselessKeyword only match whole words
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")

        fnumber = Regex(r"[+=]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")

        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        add_op = plus | minus
        mult_op = mult | div
        exp_op = Literal("^")
        comparison_op = oneOf(" ".join(self.conditionals))
        qm, colon = map(Literal, "?:")
        assignment = Literal("=")
        assignment_op = ident + assignment + ~FollowedBy(assignment)

        expr = Forward()
        expr_list = delimitedList(Group(expr))

        # Add parse action that replaces the function identifier with a (name, number of args) tuple
        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            lambda t: t.insert(0, (t.pop(0), len(t[0])))
        )
        atom = (
            add_op[...]
            + (
                (fn_call | pi | e | fnumber | ident).setParseAction(self.push_first)
                | Group(lpar + expr + rpar)
            )
        ).setParseAction(self.push_unary_minus)

        # By defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...",
        # we get right-to-left exponents, instead of left-to-right that is, 2 ^ 3 ^ 2 = 2 ^ (3 ^ 2), not (2 ^ 3) ^ 2.
        factor = Forward()
        factor <<= atom + (exp_op + factor).setParseAction(self.push_first)[...]
        term = factor + (mult_op + factor).setParseAction(self.push_first)[...]
        expr <<= term + (add_op + term).setParseAction(self.push_first)[...]
        comparison = expr + (comparison_op + expr).setParseAction(self.push_first)[...]
        ternary = comparison + (qm + expr + colon + expr).setParseAction(self.push_first)[...]
        assignment = Optional(assignment_op).setParseAction(self.push_last) + ternary

        self.bnf = assignment
    
    def eval(self, expression_string, container=None, **kwargs):
        long_kwargs = {}
        for key, value in kwargs.items():
            if not isinstance(value, str):
                long_kwargs[key] = value
                continue

            tokens = value.split(".")
            if len(tokens) == 1:
                # Assume a single node is the world matrix
                value = f"{tokens[0]}.worldMatrix[0]"
            else:
                # Turn all attribute names into long names for consistency with results in listConnections
                value = tokens[0]
                for token in tokens[1:]:
                    attr = f"{value}.{token}"
                    value += f".{cmds.attributeName(attr, long=True)}"
            long_kwargs[key] = value
        
        self.kwargs = long_kwargs

        # Reverse variable look up to write cleaner notes
        self._reverse_kwargs = {}
        for k, v in self.kwargs.items():
            self._reverse_kwargs[v] = k
        self.expression_string = expression_string
        self.expr_stack = []
        self.assignment_stack = []
        self.results = self.bnf.parseString(expression_string, True)
        self.container = cmds.container(name=container, current=True) if container else None
        self.created_nodes = {}
        stack = self.expr_stack[:] + self.assignment_stack[:]
        result = self.evaluate_stack(stack)

        if self.container:
            self.publish_container_attributes()
        return result
    
    def push_first(self, tokens):
        self.expr_stack.append(tokens[0])
    
    def push_last(self, tokens):
        for token in tokens:
            self.assignment_stack.append(token)
        
    def push_unary_minus(self, tokens):
        for token in tokens:
            if token == "-":
                self.expr_stack.append("unary -")
            else:
                break

    def evaluate_stack(self, stack):
        op, num_args = stack.pop(), 0
        if isinstance(op, tuple):
            op, num_args = op
        if op == "unary -":
            op1 = self.evaluate_stack(stack)
            return self.get_op_result(op, self.multiply, -1, op1)
        elif op == "?":
            # Ternary
            if_false = self.evaluate_stack(stack)
            if_true = self.evaluate_stack(stack)
            condition = self.evaluate_stack(stack)
            second_term = self.evaluate_stack(stack)
            first_term = self.evaluate_stack(stack)
            note = f"{first_term} {self.conditionals[condition]} {second_term} ? {if_true} : {if_false}"

            return self.get_op_result(note, self.condition, first_term, second_term, condition, if_true, if_false, op_str=note)
        elif op == ":":
            # Return the if_true statement to the ternary
            return self.evaluate_stack(stack)
        elif op in "+-*/^":
            # Operands are pushed onto the stack in reverse order
            op2 = self.evaluate_stack(stack)
            op1 = self.evaluate_stack(stack)
            return self.get_op_result(op, self.opn[op], op1, op2)
        elif op == "PI":
            return math.pi
        elif op == "E":
            return math.e
        elif op in self.functions:
            # Args are pushed onto the stack in reverse order
            args = reversed([self.evaluate_stack(stack) for _ in range(num_args)])
            args = list(args)
            return self.get_op_result(op, self.functions[op], *args)
        elif op[0].isalpha():
            value = self.kwargs.get(op)
            if value is None:
                raise Exception(f"invalid identifier '{op}'")
            return value
        elif op in self.conditionals:
            return self.conditionals.index(op)
        elif op == "=":
            destination = self.evaluate_stack(stack)
            source = self.evaluate_stack(stack)
            cmds.connectAttr(source, destination, force=True)
        else:
            # Try to evaluate as int first, then as float if int fails
            try:
                return int(op)
            except ValueError:
                return float(op)
    
    def get_op_result(self, op, func, *args, **kwargs):
        op_str = kwargs.get("op_str", self.op_str(op, *args))
        result = self.created_nodes.get(op_str)
        if result is None:
            result = func(*args)
            self.create_nodes[op_str] = result
            self.add_notes(result, op_str)
        return result
    

    def add(self, v1, v2):
        return self._connect_plus_minus_average(1, v1, v2)
    
    def subtract(self, v1, v2):
        return self._connect_plus_minus_average(2, v1, v2)
    
    def _connect_plus_minus_average(self, operation, v1, v2):
        pma = cmds.createNode("plusMinusAverage")
        cmds.setAttr(f"{pma}.operation", operation)
        in_attr = "input1D"
        out_attr = "output1D"
        # Determine whether we should use 1D or 3D attributes
        for v in [v1, v2]:
            if isinstance(v, str) and is_attribute_array(v):
                in_attr = "input3D"
                out_attr = "output3D"
        
        for i, v in enumerate([v1, v2]):
            if isinstance(v, str):
                if is_attribute_array(v):
                    cmds.connectAttr(v, f"{pma}.{in_attr}[{i}]")
                else:
                    if in_attr == "input3D":
                        for axis in "xyz":
                            cmds.connectAttr(v, f"{pma}.{in_attr}[{i}].input3D{axis}")
                    else:
                        cmds.connectAttr(v, f"{pma}.{in_attr}[{i}]")
            else:
                if in_attr == "input3D":
                    for axis in "xyz":
                        cmds.setAttr(f"{pma}.{in_attr}[{i}].input3D{axis}", v)
                else:
                    cmds.setAttr(f"{pma}.{in_attr}[{i}]", v)
        return f"{pma}.{out_attr}"

    def multiply(self, v1, v2):
        return self._connect_multiply_divide(1, v1, v2)

    def divide(self, v1, v2):
        return self._connect_multiply_divide(2, v1, v2)
    
    def pow(self, v1, v2):
        return self._connect_multiply_divide(3, v1, v2)
    
    def exp(self, v):
        return self._connect_multiply_divide(3, math.e, v)
    
    def sqrt(self, v):
        return self._connect_multiply_divide(3, v, 0.5)

    def _connect_multiply_divide(self, operation, v1, v2):
        md = cmds.createNode("multiplyDivide")
        cmds.setAttr(f"{md}.operation", operation)
        # Determine whether we should use 1D or 3D attributes
        value_count = 1
        for v in [v1, v2]:
            if isinstance(v, str) and is_attribute_array(v):
                value_count = 3
        
        for i, v in enumerate([v1, v2]):
            i += 1
            if isinstance(v, str):
                if is_attribute_array(v):
                    cmds.connectAttr(v, f"{md}.input{i}")
                else:
                    if value_count == 3:
                        for axis in "XYZ":
                            cmds.connectAttr(v, f"{md}.input{i}{axis}")
                    else:
                        cmds.connectAttr(v, f"{md}.input{i}X")
            else:
                if value_count == 3:
                    for axis in "XYZ":
                        cmds.setAttr(f"{md}.input{i}{axis}", v)
                else:
                    cmds.setAttr(f"{md}.input{i}", v)

        return f"{md}.output" if value_count == 3 else f"{md}.outputX"

    def clamp(self, value, min_value, max_value):
        clamp = cmds.createNode("clamp")

        for v, attr in [[min_value, "min"], [max_value, "max"]]:
            if isinstance(v, str):
                if is_attribute_array(v):
                    cmds.connectAttr(v, f"{clamp}.{attr}")
                else:
                    for rgb in "RGB":
                        cmds.connectAttr(v, f"{clamp}.{attr}{rgb}")
            else:
                for rgb in "RGB":
                    cmds.setAttr(f"{clamp}.{attr}{rgb}", v)
        
        value_count = 1
        if isinstance(v, str):
            if is_attribute_array(value):
                value_count = 3
                cmds.connectAttr(value, f"{clamp}.input")
            else:
                for rgb in "RGB":
                    cmds.connectAttr(value, f"{clamp}.input{rgb}")
        else:
            # Unlikely for a static value to be clamped, but it should still work
            for rgb in "RGB":
                cmds.setAttr(f"{clamp}.input{rgb}", value)
        
        return f"{clamp}.output" if value_count == 3 else f"{clamp}.outputR"

    def condition(self, first_term, second_term, operation, if_true, if_false):
        cnd = cmds.createNode("condition")
        cmds.setAttr(f"{cnd}.operation", operation)

        for v, attr in [[first_term, "firstTerm"], [second_term, "secondTerm"]]:
            if isinstance(v, str):
                cmds.connectAttr(v, f"{cnd}.{attr}")
            else:
                cmds.setAttr(f"{cnd}.{attr}", v)
        
        value_count = 1
        for v, attr in [[if_true, "colorIfTrue"], [if_false, "colorIfFalse"]]:
            if isinstance(v, str):
                if is_attribute_array(v):
                    value_count = 3
                    cmds.connectAttr(v, f"{cnd}.{attr}")
                else:
                    for rgb in "RGB":
                        cmds.connectAttr(v, "{node}.{attr}{rgb}")
            else:
                cmds.setAttr(f"{cnd}.{attr}R", v)
            
            return f"{cnd}.outColor" if value_count == 3 else f"{cnd}.outColorR"
    
    def lerp(self, a, b, t):
        bta = cmds.createNode("blendTwoAttr")

        if isinstance(t, str):
            cmds.connectAttr(t, "{bta}.attributesBlender")
        else:
            # Static value on attributesBlender doesn't make much sence
            # but we don't want to erro out
            cmds.setAttr(f"{bta}.attributesBlender", t)
        
        for i, v in enumerate([a, b]):
            if isinstance(v, str):
                cmds.connectAttr(v, f"{bta}.input[{i}]")
            else:
                cmds.setAttr(f"{bta}.input[{i}]", v)
        
        return f"{bta}.output"
    
    def abs(self, x):
        return dgexp("x > 0 ? x : -x", x=x)
    
    def min(self, x, y):
        return self.condition(x, y, self.conditionals.index("<="), x, y)
    
    def max(self, x, y):
        return self.condition(x, y, self.conditionals.index(">="), x, y)
    
    def sin(self, x):
        return self._euler_to_quat(x, "X")
    
    def cos(self, x):
        return self._euler_to_quat(x, "W")

    def _euler_to_quat(self, x, attr):
        cmds.loadPlugin("quatNodes", quiet=False)
        mdl = cmds.createNode("multDoubleLinear")
        cmds.setAttr(f"{mdl}.input1", 2 * 57.2958)  # To degree
        if isinstance(x, str):
            cmds.connectAttr(x, f"{mdl}.input2")
        else:
            cmds.setAttr(f"{mdl}.input2", x)
        
        quat = cmds.createNode("eulerToQuat")
        cmds.connectAttr(f"{mdl}.output", f"{quat}.inputRotateX")
        return f"{quat}.outputQuat.outputQuat{attr}"

    def tan(self, x):
        half_pi = math.pi * 0.5
        c = dgexp(f"{half_pi} - x", x=x)
        return dgexp("sin(x) / sin(c)", x=x, c=c)

    def acos(self, x):
        angle = cmds.createNode("angleBetween")
        for attr in [f"{i}{j}" for i in "12" for j in "XYZ"]:
            cmds.setAttr(f"{angle}.vector{attr}", 0)
        
        if isinstance(x, str):
            cmds.connectAttr(x, f"{angle}.vector1X")
            dgexp("y = x == 0.0 ? 1.0 : abs(x)", y=f"{angle}.vector2X", x=x)
        else:
            cmds.setAttr(f"{angle}.vector1X", x)
            cmds.setAttr(f"{angle}.vector2X", math.fabs(x))
        dgexp("y + sqrt(1.0 - x*x)", y=f"{angle}.vector1Y", x=x)
        return f"{angle}.axisAngle.angle"

    def asin(self, x):
        angle = cmds.createNode("angleBetween")
        for attr in [f"{i}{j}" for i in "12" for j in "XYZ"]:
            cmds.setAttr(f"{angle}.vector{attr}", 0)
        
        if isinstance(x, str):
            cmds.connectAttr(x, f"{angle}.vector1Y")
        else:
            cmds.setAttr(f"{angle}.vector1Y", x)
        result = dgexp("sqrt(1.0 - x*x)", x=x)
        cmds.connectAttr(result, f"{angle}.vector1X")
        dgexp("y = abs(x) == 1.0 ? 1.0 : r", y=f"{angle}.vector2X", x=x, r=result)
        return dgexp("x < 0 ? -y : y", x=x, y=f"{angle}.axisAngle.angle")
    
    def atan(self, x):
        angle = cmds.createNode("angleBetween")
        for attr in [f"{i}{j}" for i in "12" for j in "XYZ"]:
            cmds.setAttr(f"{angle}.vector{attr}", 0)
        cmds.setAttr(f"{angle}.vector1X", 1)
        cmds.setAttr(f"{angle}.vector2X", 1)

        if isinstance(x, str):
            cmds.connectAttr(x, f"{angle}.vector1Y")
        else:
            cmds.setAttr(f"{angle}.vector1Y", x)
        return dgexp("x < 0 ? -y : y", x=x, y=f"{angle}.axisAngle.angle")

    def distance(self, node1, node2):
        db = cmds.createNode("distanceBetween")
        cmds.connectAttr(node1, f"{db}.inMatrix1")
        cmds.connectAttr(node2, f"{db}.inMatrix2")
        return f"{db}.distance"

    def add_notes(self, node, op_str):
        node = node.split(".")[0]
        attrs = cmds.listAttr(node, userDefined=True) or []
        if "notes" not in attrs:
            cmds.addAttr(node, longName="notes", dataType="string")
        keys = self.kwargs.keys()
        keys.sort()
        notes = f"node generated by dgexp\n\n"
        notes += f"Expression:\n  {self.expression_string}\n\n"
        notes += f"Operation:\n  {op_str}\n\n"
        notes += "kwargs:\n  {}".format("\n  ".join([f"{x}: {self.kwargs[x]}" for x in keys]))
        cmds.setAttr(f"{node}.notes", notes, type="string")
    
    def publish_container_attributes(self):
        self.add_notes(self.container, self.expression_string)
        external_connections = cmds.container(self.container, q=True, connectionList=True)
        external_connections = set(external_connections)
        container_nodes = set(cmds.container(self.container, q=True, nodeList=True))
        for key, value in self.kwargs.items():
            if not isinstance(value, str):
                continue

            # To connect multiple attributes to a bound container attribute,
            # we need to create an intermediary attribute that is bound and 
            # connected to the internal attributes
            attr_type = attribute_type(value)
            kwargs = {"dt": attr_type} if attr_type == "matrix" else {"at": attr_type}
            cmds.addAttr(self.container, longName=f"_{key}", **kwargs)
            published_attr = f"{self.container}._{key}"
            cmds.container(self.container, e=True, publishAndBind=[published_attr, key])
            cmds.connectAttr(value, published_attr)

            # Reroute connections into the container to go through the published attribute
            if value in external_connections:
                connected_nodes = set(cmds.listConnections(value, source=False, plugs=True))
                for connection in connected_nodes:
                    node_name = connection.split(".")[0]
                    if node_name in container_nodes:
                        cmds.connectAttr(published_attr, connection, force=True)
                
                source_plug = cmds.listConnections(value, destination=False, plugs=True)
                if source_plug:
                    source_plug = source_plug[0]
                    node_name = source_plug.split(".")[0]
                    if node_name in container_nodes:
                        cmds.connectAttr(source_plug, published_attr, force=True)
                        cmds.connectAttr(published_attr, value, force=True)
        cmds.container(self.container, e=True, current=False)
    
    def op_str(self, op, *args):
        """Get the string form of the op and args.
        
        This is used for notes on the node as well as identifying which nodes can be reused.

        Args:
            op (str): Name of the op
            args (): Optional op arguments
        
        Returns:
            str: The uniqe op string
        """
        args = [str(v) for v in args]
        if op in self.functions:
            return f"{op}({', '.join(args)})"
        elif args:
            return op.join([self._reverse_kwargs.get(x, x) for x in args])
        return op


def is_attribute_array(value):
    return attribute_type(value) in ["double3", "float3"]


def attribute_type(attr):
    tokens = attr.split(".")
    node = tokens[0]
    attribute = tokens[-1]
    if attribute.startswith("worldMatrix"):
        # attributeQuery doesn't seem to work with worldMatrix
        return "matrix"
    return cmds.attributeQuery(attribute, node=node, at=True)
