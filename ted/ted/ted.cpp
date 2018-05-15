#include <Python.h>
#include <vector>
#include <list>
#include <stack>
#include <tuple>
#include <deque>
#include <algorithm>
#include <map>
#include <stdio.h>
#include <string.h>
#include <iostream>

using namespace std;

#define REMOVE 1
#define INSERT 1
#define UPDATE 1

class Node
{
    friend class AnnotatedTree;
    friend class Distance;

  public:
    Node(int id) : _id(id), _parent(NULL) {}
    ~Node()
    {
        for (auto n : children)
            delete n;
    }
    Node *getParent() { return _parent; }
    Node *getLastChild() { return children.empty() ? NULL : children.back(); }
    void addkid(Node *n)
    {
        children.push_back(n);
        n->_parent = this;
    }

    bool operator==(const Node &i) const { return (_id == i._id); }
    bool operator!=(const Node &i) const { return (_id != i._id); }
    friend ostream &operator<<(ostream &stream, const Node &node)
    {
        stream << node._id << endl;
        for (auto n : node.children)
        {
            stream << "  " << *n;
        }
        return stream;
    }

  protected:
    int _id;
    Node *_parent;
    vector<Node *> children;
};

class AnnotatedTree
{
    typedef deque<Node *> nodeDeque;
    typedef deque<int> intDeque;
    friend class Distance;

  public:
    AnnotatedTree() {}
    AnnotatedTree(Node *root)
    {
        stack<tuple<Node *, int, intDeque *>> pstack;
        stack<tuple<Node *, intDeque *>> nstack;
        auto rootDeque = new intDeque;
        nstack.push(make_tuple(root, rootDeque));
        int j = 0;
        while (!nstack.empty())
        {
            auto [n, anc] = nstack.top();
            nstack.pop();
            int nid = j;
            for (auto c : n->children)
            {
                auto a = new intDeque(anc->begin(), anc->end());
                a->push_front(nid);
                nstack.push(make_tuple(c, a));
            }
            pstack.push(make_tuple(n, nid, anc));
            j += 1;
        }
        map<int, int> lmds;
        map<int, int> keyroot;
        int i = 0;
        while (!pstack.empty())
        {
            auto [n, nid, anc] = pstack.top();
            pstack.pop();
            _nodes.push_back(n);
            _ids.push_back(nid);
            int lmd;
            if (n->children.empty())
            {
                lmd = i;
                for (auto a : *anc)
                {
                    if (lmds.find(a) == lmds.end())
                    {
                        lmds[a] = i;
                    }
                    else
                        break;
                }
            }
            else
            {
                lmd = lmds[nid];
            }
            _lmds.push_back(lmd);
            keyroot[lmd] = i;
            i += 1;
            delete anc;
        }
        for (auto k : keyroot)
            _keyroot.push_back(k.second);
        std::sort(_keyroot.begin(), _keyroot.end());
    }

    Node *root;
    vector<Node *> _nodes;
    vector<int> _ids;  // match list of ids
    vector<int> _lmds; // left most descendants
    vector<int> _keyroot;
};

class Distance
{

  public:
    Distance(Node *A_root, Node *B_root)
    {
        A = AnnotatedTree(A_root);
        B = AnnotatedTree(B_root);
        size_A = A._nodes.size();
        size_B = B._nodes.size();
        treedists = new float *[size_A];
        for (auto i = 0; i < size_A; i++)
            treedists[i] = new float[size_B]();
    }
    ~Distance()
    {
        for (auto i = 0; i < size_A; i++)
            delete[] treedists[i];
        delete[] treedists;
    }

    float updateCost(Node *n1, Node *n2)
    {
        if (*n1 == *n2)
        {

            return 0;
        }
        return 1;
    }

    void treedist(int i, int j)
    {
        auto Al = A._lmds;
        auto Bl = B._lmds;
        auto An = A._nodes;
        auto Bn = B._nodes;

        auto m = i - Al[i] + 2;
        auto n = j - Bl[j] + 2;
        float fd[m][n];
        memset(fd, 0, m * n * sizeof(float));

        auto ioff = Al[i] - 1;
        auto joff = Bl[j] - 1;
        for (auto x = 1; x < m; x++)
        {
            // auto node = An[x+ioff];
            fd[x][0] = fd[x - 1][0] + REMOVE;
        }
        for (auto y = 1; y < n; y++)
        {
            // auto node = Bn[y+joff];
            fd[0][y] = fd[0][y - 1] + INSERT;
        }
        for (auto x = 1; x < m; x++)
        {
            for (auto y = 1; y < n; y++)
            {
                auto node1 = An[x + ioff];
                auto node2 = Bn[y + joff];
                if (Al[i] == Al[x + ioff] and Bl[j] == Bl[y + joff])
                {
                    float costs[] = {fd[x - 1][y] + REMOVE, fd[x][y - 1] + INSERT, fd[x - 1][y - 1] + updateCost(node1, node2)};
                    fd[x][y] = *min_element(costs, costs + 3);
                    treedists[x + ioff][y + joff] = fd[x][y];
                }
                else
                {
                    auto p = Al[x + ioff] - 1 - ioff;
                    auto q = Bl[y + joff] - 1 - joff;
                    float costs[] = {fd[x - 1][y] + REMOVE, fd[x][y - 1] + INSERT, fd[p][q] + treedists[x + ioff][y + joff]};
                    fd[x][y] = *min_element(costs, costs + 3);
                }
            }
        }
    }

    float distance()
    {
        for (auto i : A._keyroot)
        {
            for (auto j : B._keyroot)
            {
                treedist(i, j);
            }
        }
        auto size_A = A._nodes.size();
        auto size_B = B._nodes.size();
        return treedists[size_A - 1][size_B - 1];
    }

    AnnotatedTree A;
    AnnotatedTree B;
    int size_A;
    int size_B;
    float **treedists;
};

static Node *
buildTree(vector<long> *seq, vector<long> *meta)
{
    Node *root = new Node(0);
    Node *curNode = root;
    long open = (*meta)[0];
    long close = (*meta)[1];
    for (auto n : *seq)
    {
        if (n == open)
            curNode = curNode->getLastChild();
        else if (n == close)
            curNode = curNode->getParent();
        else
            curNode->addkid(new Node(n));
        if (curNode == NULL)
            break;
    }
    return root;
}

static unsigned long
cted(unsigned long n)
{
    unsigned long a = 1;
    unsigned long b = 1;
    unsigned long c;

    if (n <= 1)
    {
        return 1;
    }

    while (--n > 1)
    {
        c = a + b;
        a = b;
        b = c;
    }

    return b;
}
static unsigned int
test(vector<long> *v)
{
    int ret = 0;
    for (auto a : *v)
    {
        ret += a;
    }
    return ret;
}

PyDoc_STRVAR(ted_doc, "Tree edit distance using the Zhang Shasha algorithm");

static PyObject *
pyted(PyObject *self, PyObject *args)
{
    PyObject *vector_a, *vector_b, *vector_c;

    /* get one argument as a sequence */
    if (!PyArg_ParseTuple(args, "OOO", &vector_a, &vector_b, &vector_c))
        return NULL;

    PyObject *sequence_a;
    PyObject *sequence_b;
    PyObject *sequence_c;

    long len_a, len_b, len_c;

    sequence_a = PySequence_Fast(vector_a, "expected a sequence");
    sequence_b = PySequence_Fast(vector_b, "expected a sequence");
    sequence_c = PySequence_Fast(vector_c, "expected a sequence");

    len_a = PySequence_Size(sequence_a);
    len_b = PySequence_Size(sequence_b);
    len_c = PySequence_Size(sequence_c);

    if (len_c != 4 or len_a < 0 or len_b < 0)
        return NULL;

    long array_a[len_a];
    long array_b[len_b];
    long array_c[len_c];
    vector<long> *v_a = new vector<long>;
    vector<long> *v_b = new vector<long>;
    vector<long> *v_c = new vector<long>;

    for (auto i = 0; i < len_a; i++)
    {
        PyObject *a_i = PySequence_Fast_GET_ITEM(vector_a, i);
        array_a[i] = PyLong_AsLong(a_i);
    }
    for (auto i = 0; i < len_b; i++)
    {
        PyObject *b_i = PySequence_Fast_GET_ITEM(vector_b, i);
        array_b[i] = PyLong_AsLong(b_i);
    }
    for (auto i = 0; i < len_c; i++)
    {
        PyObject *c_i = PySequence_Fast_GET_ITEM(vector_c, i);
        array_c[i] = PyLong_AsLong(c_i);
    }
    v_a->assign(array_a, array_a + len_a);
    v_b->assign(array_b, array_b + len_b);
    v_c->assign(array_c, array_c + len_c);
    Node *root_a = buildTree(v_a, v_c);
    Node *root_b = buildTree(v_b, v_c);
    Distance *d = new Distance(root_a, root_b);
    float result = d->distance();
    delete v_a;
    delete v_b;
    delete v_c;
    delete d;
    delete root_a;
    delete root_b;
    Py_BuildValue("f", result);
}

PyMethodDef methods[] = {
    {"ted", (PyCFunction)pyted, METH_VARARGS, ted_doc},
    {NULL},
};

PyDoc_STRVAR(ted_module_doc, "provides a tedonacci function");

PyModuleDef ted_module = {
    PyModuleDef_HEAD_INIT,
    "ted",
    ted_module_doc,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC
PyInit_ted(void)
{
    return PyModule_Create(&ted_module);
}
