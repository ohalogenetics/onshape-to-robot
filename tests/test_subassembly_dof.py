"""Regression tests for DOF (`dof_`) mates placed between two parts that live
inside the *same* subassembly instance.

onshape-to-robot identifies a body by the first element of a mate's occurrence
path (the top-level instance). A `dof_` mate whose two ends are both inside one
subassembly instance therefore collapses onto a single body, producing a
self-loop and "The DOF graph is not a tree". These tests pin the desired
behaviour: such a mate must split the subassembly into two bodies joined by one
joint, while a subassembly with no internal DOF stays a single rigid body.
"""

from types import SimpleNamespace

from onshape_to_robot.assembly import Assembly

I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # identity transform (row-major 4x4)
CS = {"xAxis": [1, 0, 0], "yAxis": [0, 1, 0], "zAxis": [0, 0, 1], "origin": [0, 0, 0]}


def _instance(iid, itype="Part", element="partE"):
    return {
        "id": iid,
        "name": iid,
        "type": itype,
        "documentId": "doc",
        "documentMicroversion": "mv",
        "elementId": element,
        "configuration": "default",
        "suppressed": False,
    }


def _occ(path):
    return {"path": list(path), "transform": list(I4)}


def _mate(name, mate_type, path_a, path_b):
    return {
        "featureType": "mate",
        "suppressed": False,
        "featureData": {
            "name": name,
            "mateType": mate_type,
            "matedEntities": [
                {"matedOccurrence": list(path_a), "matedCS": CS},
                {"matedOccurrence": list(path_b), "matedCS": CS},
            ],
        },
    }


def _run(assembly_data):
    """Run the occurrence->body->tree pipeline on injected assembly data."""
    a = Assembly.__new__(Assembly)
    a.config = SimpleNamespace(ignore_limits=True)
    a.assembly_data = assembly_data
    a.occurrences = {
        tuple(o["path"]): o for o in assembly_data["rootAssembly"]["occurrences"]
    }
    a.current_body_id = 0
    a.instance_body = {}
    a.dofs = []
    a.frames = []
    a.closures = []
    a.root_nodes = []
    a.tree_children = {}
    a.find_instances()
    a.process_mates()
    a.build_trees()
    return a


def _subassembly_with_internal_dof():
    """Root assembly holding one subassembly instance `S` whose two parts A and
    B are joined by a root-level `dof_slide` slider."""
    return {
        "rootAssembly": {
            "instances": [_instance("S", "Assembly", element="subE")],
            "occurrences": [_occ(["S"]), _occ(["S", "A"]), _occ(["S", "B"])],
            "features": [_mate("dof_slide", "SLIDER", ["S", "A"], ["S", "B"])],
        },
        "subAssemblies": [
            {
                "documentId": "doc",
                "documentMicroversion": "mv",
                "elementId": "subE",
                "configuration": "default",
                "instances": [_instance("A"), _instance("B")],
                "features": [],
            }
        ],
    }


def test_dof_between_two_parts_in_one_subassembly_builds_a_tree():
    a = _run(_subassembly_with_internal_dof())

    # The joint must connect two *distinct* bodies (not a self-loop).
    assert len(a.dofs) == 1
    assert a.dofs[0].body1_id != a.dofs[0].body2_id
    # Exactly two rigid bodies participate.
    assert len({a.dofs[0].body1_id, a.dofs[0].body2_id}) == 2


def test_split_bodies_expose_their_own_part_occurrence():
    """Each side of the split subassembly maps back to exactly its own part."""
    a = _run(_subassembly_with_internal_dof())
    b1, b2 = a.dofs[0].body1_id, a.dofs[0].body2_id
    paths1 = {tuple(o["path"]) for o in a.body_occurrences(b1)}
    paths2 = {tuple(o["path"]) for o in a.body_occurrences(b2)}
    assert paths1 == {("S", "A")} or paths1 == {("S", "B")}
    assert paths2 == {("S", "A")} or paths2 == {("S", "B")}
    assert paths1 != paths2


def _subassembly_without_dof():
    """Subassembly `S` (two rigid parts A, B) joined to a top-level part `P` by
    one root-level slider. The subassembly has no internal DOF."""
    return {
        "rootAssembly": {
            "instances": [
                _instance("S", "Assembly", element="subE"),
                _instance("P"),
            ],
            "occurrences": [
                _occ(["S"]),
                _occ(["S", "A"]),
                _occ(["S", "B"]),
                _occ(["P"]),
            ],
            "features": [_mate("dof_slide", "SLIDER", ["S", "A"], ["P"])],
        },
        "subAssemblies": [
            {
                "documentId": "doc",
                "documentMicroversion": "mv",
                "elementId": "subE",
                "configuration": "default",
                "instances": [_instance("A"), _instance("B")],
                "features": [],
            }
        ],
    }


def test_subassembly_without_internal_dof_stays_one_rigid_body():
    """Regression guard: a subassembly with no internal DOF is still collapsed
    to a single rigid body (the default behaviour must not change)."""
    a = _run(_subassembly_without_dof())
    # The subassembly is one rigid body keyed by its top-level instance "S";
    # its internal parts are not split into full-path bodies.
    assert "S" in a.instance_body
    assert ("S", "A") not in a.instance_body
    assert ("S", "B") not in a.instance_body
    # The dof joins the (single) subassembly body to P — two bodies, one joint.
    assert len(a.dofs) == 1
    assert a.dofs[0].body1_id != a.dofs[0].body2_id
    assert a.instance_body["S"] in {a.dofs[0].body1_id, a.dofs[0].body2_id}
