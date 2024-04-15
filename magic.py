#region
from OpenGL.GL import *
import glfw

import numpy as np
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at
import math
import sys

import imgui

from collections.abc import Callable

# we use 'warnings' to remove this warning that ImGui[glfw] gives
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from imgui.integrations.glfw import GlfwRenderer as ImGuiGlfwRenderer

from lab_utils import Vec3, Mat3, Mat4, make_translation, normalize
#endregion
#region
g_mousePos = [0.0, 0.0]

VAL_Position = 0
g_vertexDataBuffer = 0
g_vertexArrayObject = 0
g_simpleShader = 0
#endregion
#region
def get_shader_info_log(obj: int) -> str:
    """
        Get the current error message from the shader.

        Parameters:

            obj: integer handle to the shader object being
            compiled or linked.
        
        Returns:

            The current error message, or an empty string
            if the operation failed silently (GOOD LUCK)
    """
    logLength = glGetShaderiv(obj, GL_INFO_LOG_LENGTH)

    if logLength > 0:
        return glGetShaderInfoLog(obj).decode()

    return ""

def compile_and_attach_shader_module(shader_program: int, 
    shader_stage: int, source_code: str) -> bool:
    """
        Compile the source code for a shader module 
        (e.g., vertex / fragment) and attaches it to the
        given shader program.

        Parameters:

            shader_program: the program to attach the compiled
            module to.

            shader_stage: the stage which the module is meant for

            source_code: the source code to be compiled.
        
        Returns:

            Whether the source code was successfully compiled.
    """

    # Create the opengl shader module object
    module = glCreateShader(shader_stage)
    # upload the source code for the shader
    # Note the function takes an array of source strings and lengths.
    glShaderSource(module, [source_code])
    glCompileShader(module)

    """
        If there is a syntax or other compiler error during shader 
        compilation, we'd like to know
    """
    compile_ok = glGetShaderiv(module, GL_COMPILE_STATUS)

    if not compile_ok:
        err = get_shader_info_log(module)
        print("SHADER COMPILE ERROR: '%s'" % err);
        return False

    glAttachShader(shader_program, module)
    glDeleteShader(module)
    return True

def debug_message_callback(
    source: int, type: int, id: int, 
    severity: int, length: int, message: str, 
    userParam: c_void_p) -> None:
    print(message)

def build_basic_shader(
    vertex_shader_source: str, fragment_shader_source: int) -> int:
    """
        creates a basic shader that binds the 0th attribute stream 
        to the shader attribute "positionIn" and the output shader 
        variable 'fragmentColor' to the 0th render target 
        (the default)

        Parameters:

            vertex_shader_source, fragment_shader_source: source code
                for the vertex and fragment shader modules.
            
        Returns:

            Integer handle to the created shader program.
    """
    
    shader = glCreateProgram()

    if compile_and_attach_shader_module(
        shader, GL_VERTEX_SHADER, vertex_shader_source) \
        and compile_and_attach_shader_module(
            shader, GL_FRAGMENT_SHADER, fragment_shader_source):
        """
	        Link the name we used in the vertex shader 'positionIn' 
            to the integer index 0. This ensures that when the shader 
            executes, data fed into 'positionIn' will be sourced from 
            the 0'th generic attribute stream. This seemingly 
            backwards way of telling the shader where to look allows 
            OpenGL programs to swap vertex buffers without needing 
            to do any string lookups at run time.
        """
        glBindAttribLocation(shader, 0, "positionIn")

        """
	        If we have multiple images bound as render targets, 
            we need to specify which 'out' variable in the fragment 
            shader goes wher. In this case it is totally redundant 
            as we only have one (the default render target, or 
            frame buffer) and the default binding is always zero.
        """
        glBindFragDataLocation(shader, 0, "fragmentColor")

        """
            once the bindings are done we can link the program stages
            to get a complete shader pipeline. This can yield errors, 
            for example if the vertex and fragment modules don't have 
            compatible out and in variables (e.g., the fragment 
            module expects some data that the vertex module is not 
            outputting).
        """
        glLinkProgram(shader)
        linkStatus = glGetProgramiv(shader, GL_LINK_STATUS)
        if not linkStatus:
            err = glGetProgramInfoLog(shader)
            print("SHADER LINKER ERROR: '%s'" % err)
            sys.exit(1)
    return shader

def flatten(array: list[list[float]]) -> np.ndarray:
    """
        Turns a multidimensional array into a 1D array

        Parameters:

            array: the array to be flattened

        Returns:

            a one dimensional numpy array holding the
            original data.
    """
    data_array = np.array(array, dtype = np.float32)
    length = data_array.nbytes // data_array.itemsize
    return data_array.reshape(length)

def upload_float_data(
    buffer_object: int, float_data: list[list[float]]) -> None:
    """
        Uploads the given set of floats to the given buffer object.

        Parameters:

            buffer_object: integer handle to the buffer which
                will be written to.
            
            float_data: the data to be uploaded.
    """
    flat_data = flatten(float_data)
    """
        Upload data to the currently bound GL_ARRAY_BUFFER, 
        note that this is completely anonymous binary data, 
        no type information is retained (we'll supply that 
        later in glVertexAttribPointer)
    """
    glBindBuffer(GL_ARRAY_BUFFER, buffer_object)
    glBufferData(GL_ARRAY_BUFFER, flat_data.nbytes, flat_data, GL_STATIC_DRAW)

def create_vertex_array_object(vertex_positions: list[Vec3]) -> tuple[int]:
    """
        Create a vertex array object (mesh) from the given set of
        points.

        Parameters:

            vertex_positions: the list of points with which
                construct the mesh. Interpreted as positions.
        
        Returns:

            integer handles to the created vertex buffer and 
            vertex array objects.

            eg. vao, vbo = create_vertex_array_object(my_positions)
    """

    """
        GLuint &positionBuffer, GLuint &vertexArrayObject
        glGenX(<count>, <array of GLuint>) 
        is the typical pattern for creating objects in OpenGL.  
        Do pay attention to this idiosyncrasy as the first parameter 
        indicates the number of objects we want created.  
        Usually this is just one, but if you were to change the below
        code to '2' OpenGL would happily overwrite whatever is after
        'positionBuffer' on the stack (this leads to nasty bugs 
        that are sometimes very hard to detect 
        - i.e., this was a poor design choice!)
    """
    positionBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer)

    # re-package python data into something we can send to OpenGL
    flat_data = flatten(vertex_positions)
    """
        Upload data to the currently bound GL_ARRAY_BUFFER, 
        note that this is completely anonymous binary data, 
        no type information is retained 
        (we'll supply that later in glVertexAttribPointer)
    """
    glBufferData(GL_ARRAY_BUFFER, flat_data.nbytes, flat_data, GL_STATIC_DRAW)

    vertexArrayObject = glGenVertexArrays(1)
    glBindVertexArray(vertexArrayObject)
    """
        The positionBuffer is already bound to the GL_ARRAY_BUFFER 
        location. This is typical OpenGL style - you bind the buffer 
        to GL_ARRAY_BUFFER, and the vertex array object using 
        'glBindVertexArray', and then glVertexAttribPointer 
        implicitly uses both of these.  You often need to read the 
        manual or find example code. 'VAL_Position' is an integer, 
        which tells it which attribute we want to attach this data to, 
        this must be the same that we set up our shader using 
        glBindAttribLocation.  Next provide we type information 
        about the data in the buffer: 
            there are three components (x,y,z) per element (position)
            
            and they are of type 'float', and are not normalized
            
            The last arguments can be used to describe the layout 
            in more detail (stride & offset).
    
            Note: The last argument is 'pointer' and has type 
            'const void *', however, in modern OpenGL, 
            the data ALWAYS comes from the current GL_ARRAY_BUFFER 
            object, and 'pointer' is interpreted as an offset 
            (which is somewhat clumsy).
    """
    attribute_index = VAL_Position
    element_count = 3
    element_format = GL_FLOAT
    normalized = GL_FALSE
    stride = 12 #alternatively 0 can be used to indicate tightly packed data
    offset = c_void_p(0)
    glVertexAttribPointer(
        attribute_index, element_count, element_format, 
        normalized, stride, offset)
    """
        For the currently bound vertex array object, 
        enable the VAL_Position'th vertex array 
        (otherwise the data is not fed to the shader)
    """
    glEnableVertexAttribArray(attribute_index)

    """
        Unbind the buffers again to avoid unintentianal 
        GL state corruption (this is something that can 
        be rather inconventient to debug)
    """
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return (positionBuffer, vertexArrayObject)

def draw_vertex_data_as_triangles(
        triangle_vertices: list[Vec3]) -> None:
    """
        Draw the given set of data as triangles.

        Parameters:

            triangle_vertices: a list of vec3s, representing
                positions in (x,y,z) form.
    """
    global g_simpleShader
    global g_vertexArrayObject
    global g_vertexDataBuffer

    # Bind ('use') current shader program
    glUseProgram(g_simpleShader)


    upload_float_data(g_vertexDataBuffer, triangle_vertices)

    """
        The Vertex Array Object is a wrapper for both the 
        data and its attributes. To ready a mesh for drawing,
        simply bind its VAO!
    """
    glBindVertexArray(g_vertexArrayObject)

    """
        https:#www.khronos.org/registry/OpenGL-Refpages/gl4/html/glDrawArrays.xhtml
        Tell OpenGL to draw triangles using data from the currently 
        bound vertex array object by grabbing three at a time vertices,
    
        for(int i = 0; i < g_numSphereVerts; i += 3) ...  draw triangle ...
    """
    glDrawArrays(GL_TRIANGLES, 0, len(triangle_vertices))

    """
        Unbind the shader program & vertex array object to ensure it 
        does not affect anything else (in this simple program, no 
        great risk, but otherwise it pays to be careful)
    """
    glBindVertexArray(0)
    glUseProgram(0)

g_userShader = 0
g_vertexShaderSourceCode = ""

def draw_vertex_data_as_triangles_with_vertex_shader(
    triangle_vertices: list[Vec3], transform: Mat4, 
    vertex_shader_source: str) -> None:
    """
        Draw vertex data with a transform and custom shader source
        code.

        Parameters:

            triangle_vertices: list of positions to draw.

            transform: transform matrix to apply.
            
            vertex_shader_source: source code for the vertex module.
                The program will store and compare with the last
                compiled shader, so it's ok to pass the same source
                code in many times (if you like string comparisons).

    """

    global g_userShader
    global g_vertexArrayObject
    global g_vertexDataBuffer
    global g_vertexShaderSourceCode

    """
        We re-compile and create the shader program 
        if the source code has changed, this is useful for debugging,
        for example to re-load shaders from files without re-starting,
        but care must be taken that all uniforms etc are set!
    """
    if len(vertex_shader_source) != 0 \
        and g_vertexShaderSourceCode != vertex_shader_source:
        if g_userShader != 0:
            glDeleteShader(g_userShader)
        g_vertexShaderSourceCode = vertex_shader_source
        g_userShader = build_basic_shader(vertex_shader_source, """
        #version 330
        out vec4 fragmentColor;

        void main() 
        {
	        fragmentColor = vec4(1.0);
        }
        """)

    """
        Fetch the location of the uniform from the shader program 
        object - usually this is done once and then the integer 
        index can be used to set the value of the uniform in the 
        rendering loop, here we re-do this every iteration 
        for illustration.
        https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetUniformLocation.xhtml
    """
    transformUniformIndex = glGetUniformLocation(g_userShader, "transformationMatrix")

    """
        Bind ('use') current shader program
        NOTE: glGetUniformLocation takes the shader program as an 
            argument, so it does not require the shader to be bound.
            However, glUniformMatrix4fv DOES NOT take the shader 
            program as an argument and therefore it works on the one
            that is currenly bound using glUseProgram. 
            This is easy to mess up, so be careful!
    """
    glUseProgram(g_userShader)
    
    """
        Now we set the uniform to the value of the transform matrix, 
        there are functions for different types.
        NOTE: the third argument tells OpenGL whether it should 
            transpose the matrix, OpenGL expects it in 
            column-major order. This is one of those pesky details 
            that is easy to get wrong and somewhat hard to debug.
        https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glUniform.xhtml
    """
    glUniformMatrix4fv(transformUniformIndex, 1, GL_TRUE, transform.getData())

    # as before pipe up the vertex data.
    upload_float_data(g_vertexDataBuffer, triangle_vertices)

    # Bind gl object storing the mesh data
    # https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBindVertexArray.xhtml
    glBindVertexArray(g_vertexArrayObject)

    """
        Tell OpenGL to draw triangles using data from the currently 
        bound vertex array object by grabbing vertices three at a time
        up to len(triangleVerts) vertices, 
        something like (but in hardware)
        for(i in range(0, len(triangleVerts), 3): ...  draw triangle ...
        https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glDrawArrays.xhtml
    """
    glDrawArrays(GL_TRIANGLES, 0, len(triangle_vertices))

    """
        Unbind the shader program & vertex array object to 
        ensure it does not affect anything else 
        (in this simple program, no great risk, but otherwise 
        it pays to be careful)
    """
    glBindVertexArray(0)
    glUseProgram(0)

def begin_imgui_hud() -> None:
    """
        Begin an imgui renderpass.
    """
    global g_mousePos
    #imgui.set_next_window_position([1.0,1.0], imgui.ALWAYS, [float(imgui.get_io().display_size.x) - 5.0, 5.0])
    #imgui.set_next_window_position([1.0,1.0], imgui.ALWAYS, [float(imgui.get_io().display_size.x) - 5.0, 5.0])
    #imgui.set_next_window_position(1.0,1.0)
    imgui.set_next_window_position(5.0, 5.0)
    #imgui.setpi
    #imgui.set_next_window_size(float(imgui.get_io().display_size.x) - 5.0, 5.0)
#, imgui.ALWAYS, [])

    #imgui.SetNextWindowBgAlpha(0.5f); // Transparent background
    imgui_flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE\
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_MOVE \
        | imgui.WINDOW_NO_SAVED_SETTINGS\
        | imgui.WINDOW_NO_FOCUS_ON_APPEARING
    if imgui.begin("Example: Fixed Overlay", 0, imgui_flags):
        imgui.text("Mouse Position: {%0.3f, %0.3f}" %(g_mousePos[0], g_mousePos[1]))

def end_imgui_hud() -> None:
    """
        End an imgui renderpass.
    """
    
    imgui.end()

def run_program(title: str, 
    start_width: int, start_height: int, 
    render_frame: Callable[[int, int], None], 
    init_resources: Callable[[], None] | None = None, 
    draw_ui: Callable[[], None] | None = None):
    """
        Start the program.

        Parameters:

            start_width, start_height: the size of the framebuffer/window

            render_frame: function to be called every frame to draw
                everything.
            
            init_resources: optional function to be called upon
                OpenGL context creation.
            
            draw_ui: optional function to be called every frame
                during the imgui renderpass.
    """

    global g_simpleShader
    global g_vertexArrayObject
    global g_vertexDataBuffer
    global g_mousePos

    if not glfw.init():
        sys.exit(1)

    #glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, 1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)


    window = glfw.create_window(start_width, start_height, title, None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window) 

    print("--------------------------------------\nOpenGL\n  Vendor: %s\n  Renderer: %s\n  Version: %s\n--------------------------------------\n" % (glGetString(GL_VENDOR).decode("utf8"), glGetString(GL_RENDERER).decode("utf8"), glGetString(GL_VERSION).decode("utf8")), flush=True)

    imgui.create_context()
    impl = ImGuiGlfwRenderer(window)

    #region
    #For OpenGL 4.3 and beyond:
    #glDebugMessageCallback(GLDEBUGPROC(debug_message_callback), None)

    #glEnable(GL_DEBUG_OUTPUT)
    """
        This ensures that the callback is done in the context of the 
        calling function, which means it will be on the stack in the 
        debugger, which makes it a lot easier to figure out why 
        it happened.
    """
    #glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS)
    #endregion

    g_simpleShader = build_basic_shader(
    """
    #version 330
    in vec3 positionIn;

    void main() 
    {
	    gl_Position = vec4(positionIn, 1.0);
    }
    """,
    """
    #version 330
    out vec4 fragmentColor;

    void main() 
    {
	    fragmentColor = vec4(1.0);
    }
    """)

    # Create Vertex array object and buffer with dummy data for now, we'll fill it later when rendering the frame
    (g_vertexDataBuffer, g_vertexArrayObject) = create_vertex_array_object([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    glDisable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    if init_resources:
        init_resources()

    while not glfw.window_should_close(window):
        
        # Poll for and process events
        glfw.poll_events()
        impl.process_inputs()
        
        # Render here, e.g.  using pyOpenGL
        width, height = glfw.get_framebuffer_size(window)
    
        mouseX,mouseY = glfw.get_cursor_pos(window)
        g_mousePos = [mouseX,mouseY]
        
        imgui.new_frame()

        begin_imgui_hud()

        render_frame(width, height)
    
        #imgui.show_test_window()
        if draw_ui:
            draw_ui()

        end_imgui_hud()
        imgui.render()
        impl.render(imgui.get_draw_data())
        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()

def build_shader(
    vertex_shader_source: str, fragment_shader_source: str, 
    attrib_locations: dict[str, int], 
    frag_data_locations: dict[str, int] = {}) -> int:
    """
        Creates a more general shader that binds a map of attribute 
        streams to the shader and the also any number of output 
        shader variables.
        The fragDataLocs can be left out for programs that don't 
        use multiple render targets as the default for any 
        output variable is zero.

        Parameters:

            vertex_shader_source, fragment_shader_source: source code
                for the vertex and fragment modules.
            
            attrib_locations: location of each attribute.
                eg. {"position": 0, "colour": 1, ...}
            
            frag_data_locations: optional, describes each colour
                attachment.

                eg. {"albedo": 0, "normal": 1, ...}
    """
    shader = glCreateProgram()

    if compile_and_attach_shader_module(
        shader, GL_VERTEX_SHADER, vertex_shader_source) \
        and compile_and_attach_shader_module(
            shader, GL_FRAGMENT_SHADER, fragment_shader_source):
	    # Link the attribute names we used in the vertex shader to the integer index
        for name, location in attrib_locations.items():
            glBindAttribLocation(shader, location, name)

        """
	        If we have multiple images bound as render targets, 
            we need to specify which 'out' variable in the fragment 
            shader goes where in this case it is totally redundant 
            as we only have one (the default render target, 
            or frame buffer) and the default binding is always zero.
        """
        for name, location in frag_data_locations.items():
            glBindFragDataLocation(shader, location, name)

        """
            once the bindings are done we can link the program stages 
            to get a complete shader pipeline. This can yield errors,
            for example if the vertex and fragment shaders don't have 
            compatible out and in variables (e.g., the fragment 
            shader expects some data that the vertex shader is not 
            outputting).
        """
        glLinkProgram(shader)
        linkStatus = glGetProgramiv(shader, GL_LINK_STATUS)
        if not linkStatus:
            err = glGetProgramInfoLog(shader)
            print("SHADER LINKER ERROR: '%s'" % err)
            sys.exit(1)
    return shader

def make_lookAt(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Mat4:
    """
        Constructs and returns a 4x4 matrix representing a 
        view transform, i.e., from world to view space.
        this is basically the same as what we saw in 
        Lecture #2 for placing the car in the world, 
        except the inverse! (and also view-space 'forwards' is the 
        negative z-axis)

        Parameters:

            eye: The camera's position

            target: Point being looked at

            up: The camera's up vector (helps to orient the camera).
        
        Returns:

            An appropriate view transform.
    """

    #forwards
    F = np.array(target[:3]) - np.array(eye[:3])
    f = normalize(F)

    #up
    U = np.array(up[:3])

    #side
    s = normalize(np.cross(f, U))

    u = normalize(np.cross(s, f))

    """
        {side, up, forwards} now form an orthonormal
        basis for R3, being unit vectors in the camera's
        local {x,y,z} axes respectively.

        Now we use this to produce the affine inverse of
        the camera's frame of reference.
    """

    M = np.array((
        (           s[0],            u[0],          -f[0], 0.0),
        (           s[1],            u[1],          -f[1], 0.0),
        (           s[2],            u[2],          -f[2], 0.0),
        (-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye), 1.0)), 
        dtype=np.float32)
    return Mat4(M)

def make_perspective(
    fovy: float, aspect: float, n: float, f: float) -> Mat4:
    """
        Make a perspective projection matrix.

        Parameters:

            fovy: field of view (in degrees)

            aspect: aspect ratio of the screen (w/h)

            n: near distance

            f: far distance
        
        Returns:

            The perspective projection matrix.
    """
    radFovY = math.radians(fovy)
    tanHalfFovY = math.tan(radFovY / 2.0)
    sx = 1.0 / (tanHalfFovY * aspect)
    sy = 1.0 / tanHalfFovY
    zz = -(f + n) / (f - n)
    zw = -(2.0 * f * n) / (f - n)

    return Mat4([[sx,  0,  0,  0],
                 [ 0, sy,  0,  0],
                 [ 0,  0, zz, -1],
                 [ 0,  0, zw,  0]])

def get_uniform_location_debug(shader_program: int, name: str) -> int:
    """
        Attempt to fetch the location of the given uniform within the
        given shader program. Prints out a helpful message upon failure.
    """
    loc = glGetUniformLocation(shader_program, name)
    # Useful point for debugging, replace with silencable logging 
    #if loc == -1:
    #    print(f"Uniform \'{name}\' was not found")
    return loc
#endregion