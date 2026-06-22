#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>


static PyObject *
mse(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy;
    char *cx, *cy;
    double *x, *y;
    double a, b, c;
    double r;
    double p;
    bool success;

    PyArrayObject *ops[2];
    npy_uint32 op_flags[2];
    NpyIter *i;
    NpyIter_IterNextFunc* inext;
    char **idata;
    npy_intp *istride, *iinnersize;
    npy_intp icount;

    if (!PyArg_ParseTuple(args, "OOddd", &ox, &oy, &a, &b, &c)) {
        //PyErr_SetString(PyExc_Exception, "Incorrect input arguments.");
        return NULL;
    }

    // Assumptions to be ensured by Python code:
    //  x and y are arrays
    //  same length, dimension 1
    //  float64 type
    // order is C
    if (!PyArray_Check(ox)) {
        PyErr_SetString(PyExc_TypeError, "x should be a numpy array");
        goto fail;
    }
    if (!PyArray_Check(oy)) {
        PyErr_SetString(PyExc_TypeError, "y should be a numpy array");
        goto fail;
    }

    // Cast
    ops[0] = (PyArrayObject*)ox;
    ops[1] = (PyArrayObject*)oy;
    op_flags[0] = NPY_ITER_READONLY;
    op_flags[1] = NPY_ITER_READONLY;

    // Create iterator
    i = NpyIter_MultiNew(2, ops,
        NPY_ITER_EXTERNAL_LOOP | NPY_ITER_ZEROSIZE_OK,
        NPY_KEEPORDER,
        NPY_NO_CASTING,
        op_flags,
        NULL);
    if(i == NULL) {
        // Nothing allocated yet, exception is set by MultiNew, so return
        return NULL;
    }

    // Pointers to iterator functions and data
    inext = NpyIter_GetIterNext(i, NULL);
    idata = NpyIter_GetDataPtrArray(i);
    istride = NpyIter_GetInnerStrideArray(i);
    iinnersize = NpyIter_GetInnerLoopSizePtr(i);

    // Calculate
    r = 0.0;
    do {
        // Data pointers for current block
        cx = idata[0];
        cy = idata[1];

        // Inner loop over current block
        icount = *iinnersize;

        while (icount--) {
            //*data_c = *data_a + *data_b;
            x = (double *)cx;
            y = (double *)cy;

            p = a - *y + b * exp(c * *x);
            r += p * p;

            // Update data pointers
            cx += istride[0];
            cy += istride[1];
        }

    } while (inext(i));
    r /= (double)(PyArray_SIZE(ops[0]));

    /* Finished succesfully, free memory and return */
    success = 1;
fail:
    /* Free memory */
    NpyIter_Deallocate(i);

    /* Return */
    if (success) {
        return PyFloat_FromDouble(r);
    } else {
        return 0;
    }
}

static PyMethodDef cerr_module_methods[] = {
    {"mse",  mse, METH_VARARGS, "Testing an MSE in C."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// Module "execution" slot - second phase on initialisation
static bool initialised = false;
static int
cerr_module_exec(PyObject *m)
{
    if (initialised) {
        PyErr_SetString(PyExc_ImportError, "Cannot initialize expfit._cerr module more than once");
        return -1;
    }
    initialised = true;

    return 0;
}

// List of slot definitions for multi-phase initialization
static PyModuleDef_Slot cerr_module_slots[] = {
    // Not useing Py_mod_create slot
    {Py_mod_exec, cerr_module_exec},
    {0, NULL}
};

// Module definition
static struct PyModuleDef cerr_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_cerr",
    .m_size = 0, // Assume module has no state
    .m_slots = cerr_module_slots,
    .m_methods = cerr_module_methods
};

// Initialisation function
PyMODINIT_FUNC
PyInit__cerr(void)
{
    // Initialise numpy
    // Note: This is a macro that will set an error and return NULL if
    // an error occurs
    import_array();

    return PyModuleDef_Init(&cerr_module);
}
