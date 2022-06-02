use crate::simplex::Simplex;
use crate::types::{Dim, Index};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::iter;
use std::ops::Mul;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Shape {
    Point,
    Simplex(Simplex),
    Other,
}

trait SequenceTransformation: Clone {
    /// Returns the difference between the dimension of the input and the output sequence.
    fn delta_dim(&self) -> Dim;
    /// Returns the length of the output sequence given the length of the input sequence.
    fn len(&self, parent_len: Index) -> Index;
    /// Increment the offset of the coordinate transformation, if applicable.
    fn increment_offset(&mut self, amount: Dim);
    /// Increment the offset of the coordinate transformation, if applicable.
    fn decrement_offset(&mut self, amount: Dim);
    /// Map the index and coordinate of an element in the output sequence to
    /// the input sequence. The index is returned, the coordinate is adjusted
    /// in-place. If the coordinate dimension of the input sequence is larger
    /// than that of the output sequence, the [Operator::delta_dim()] last
    /// elements of the coordinate are discarded.
    fn apply_inplace(&self, index: Index, coordinate: &mut [f64]) -> Index;
    /// Map the index and multiple coordinates of an element in the output
    /// sequence to the input sequence. The index is returned, the coordinates
    /// are adjusted in-place.
    fn apply_many_inplace(&self, index: Index, coordinates: &mut [f64], dim: Dim) -> Index {
        let dim = dim as usize;
        let mut result_index = 0;
        for i in 0..coordinates.len() / dim {
            result_index = self.apply_inplace(index, &mut coordinates[i * dim..(i + 1) * dim]);
        }
        result_index
    }
    /// Map the index and coordinate of an element in the output sequence to
    /// the input sequence.
    fn apply(&self, index: Index, coordinate: &[f64]) -> (Index, Vec<f64>) {
        let delta_dim = self.delta_dim() as usize;
        let to_dim = coordinate.len() + delta_dim;
        let mut result = Vec::with_capacity(to_dim);
        result.extend_from_slice(coordinate);
        result.extend(iter::repeat(0.0).take(delta_dim));
        (self.apply_inplace(index, &mut result), result)
    }
    /// Map the index and multiple coordinates of an element in the output
    /// sequence to the input sequence.
    fn apply_many(&self, index: Index, coordinates: &[f64], dim: Dim) -> (Index, Vec<f64>) {
        assert_eq!(coordinates.len() % dim as usize, 0);
        let ncoords = coordinates.len() / dim as usize;
        let delta_dim = self.delta_dim();
        let to_dim = dim + delta_dim;
        let mut result = Vec::with_capacity(ncoords * to_dim as usize);
        for coord in coordinates.chunks(dim as usize) {
            result.extend_from_slice(&coord);
            result.extend(iter::repeat(0.0).take(delta_dim as usize));
        }
        (self.apply_many_inplace(index, &mut result, to_dim), result)
    }
    //fn shape(&self, shape: Shape, offset: Dim) -> (Shape, Dim);
}

#[derive(Debug, Clone, PartialEq)]
enum OperatorKind {
    Index(Index, Index),
    Coordinate(Dim, Dim, Dim, Index),
}

trait DescribeOperator {
    /// Returns the kind of operation the operator applies to its parent sequence.
    fn operator_kind(&self) -> OperatorKind;
    #[inline]
    fn as_children(&self) -> Option<&Children> {
        None
    }
    #[inline]
    fn as_children_mut(&mut self) -> Option<&mut Children> {
        None
    }
    #[inline]
    fn as_edges(&self) -> Option<&Edges> {
        None
    }
    #[inline]
    fn as_edges_mut(&mut self) -> Option<&mut Edges> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transpose {
    len1: Index,
    len2: Index,
}

impl Transpose {
    #[inline]
    pub fn new(len1: Index, len2: Index) -> Self {
        Self { len1, len2 }
    }
}

impl SequenceTransformation for Transpose {
    #[inline]
    fn delta_dim(&self) -> Dim {
        0
    }
    #[inline]
    fn len(&self, parent_len: Index) -> Index {
        parent_len
    }
    #[inline]
    fn increment_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn decrement_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn apply_inplace(&self, index: Index, _coordinate: &mut [f64]) -> Index {
        let low2 = index % self.len2;
        let low1 = (index / self.len2) % self.len1;
        let high = index / (self.len1 * self.len2);
        high * (self.len1 * self.len2) + low2 * self.len1 + low1
    }
}

impl DescribeOperator for Transpose {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Index(self.len1 * self.len2, self.len1 * self.len2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Take {
    indices: Box<[Index]>,
    len: Index,
}

impl Take {
    #[inline]
    pub fn new(indices: impl Into<Box<[Index]>>, len: Index) -> Self {
        Self {
            indices: indices.into(),
            len,
        }
    }
}

impl SequenceTransformation for Take {
    #[inline]
    fn delta_dim(&self) -> Dim {
        0
    }
    #[inline]
    fn len(&self, parent_len: Index) -> Index {
        (parent_len / self.len) * self.indices.len() as Index
    }
    #[inline]
    fn increment_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn decrement_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn apply_inplace(&self, index: Index, _coordinate: &mut [f64]) -> Index {
        let nindices = self.indices.len() as Index;
        let low = index % nindices;
        let high = index / nindices;
        high * self.len + self.indices[low as usize]
    }
}

impl DescribeOperator for Take {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Index(self.len, self.indices.len() as Index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Children {
    simplex: Simplex,
    offset: Dim,
}

impl Children {
    #[inline]
    pub fn new(simplex: Simplex, offset: Dim) -> Self {
        Self { simplex, offset }
    }
}

impl SequenceTransformation for Children {
    #[inline]
    fn delta_dim(&self) -> Dim {
        0
    }
    #[inline]
    fn len(&self, parent_len: Index) -> Index {
        parent_len * self.simplex.nchildren()
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        self.offset += amount;
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        self.offset -= amount;
    }
    #[inline]
    fn apply_inplace(&self, index: Index, coordinate: &mut [f64]) -> Index {
        self.simplex
            .apply_child_inplace(index, &mut coordinate[self.offset as usize..])
    }
}

impl DescribeOperator for Children {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Coordinate(
            self.offset,
            self.simplex.dim(),
            self.simplex.dim(),
            self.simplex.nchildren(),
        )
    }
    #[inline]
    fn as_children(&self) -> Option<&Children> {
        Some(self)
    }
    #[inline]
    fn as_children_mut(&mut self) -> Option<&mut Children> {
        Some(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edges {
    simplex: Simplex,
    offset: Dim,
}

impl Edges {
    #[inline]
    pub fn new(simplex: Simplex, offset: Dim) -> Self {
        Self { simplex, offset }
    }
}

impl SequenceTransformation for Edges {
    #[inline]
    fn delta_dim(&self) -> Dim {
        1
    }
    #[inline]
    fn len(&self, parent_len: Index) -> Index {
        parent_len * self.simplex.nedges()
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        self.offset += amount;
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        self.offset -= amount;
    }
    #[inline]
    fn apply_inplace(&self, index: Index, coordinate: &mut [f64]) -> Index {
        self.simplex
            .apply_edge_inplace(index, &mut coordinate[self.offset as usize..])
    }
}

impl DescribeOperator for Edges {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Coordinate(
            self.offset,
            self.simplex.dim(),
            self.simplex.edge_dim(),
            self.simplex.nedges(),
        )
    }
    #[inline]
    fn as_edges(&self) -> Option<&Edges> {
        Some(self)
    }
    #[inline]
    fn as_edges_mut(&mut self) -> Option<&mut Edges> {
        Some(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
struct FiniteF64(pub f64);

impl Eq for FiniteF64 {}

impl Ord for FiniteF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        if let Some(ord) = self.0.partial_cmp(&other.0) {
            ord
        } else {
            panic!("not finite");
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniformPoints {
    points: Box<[FiniteF64]>,
    point_dim: Dim,
    offset: Dim,
}

impl UniformPoints {
    pub fn new(points: Box<[f64]>, point_dim: Dim, offset: Dim) -> Self {
        // TODO: assert that the points are actually finite.
        let points: Box<[FiniteF64]> = unsafe { std::mem::transmute(points) };
        Self {
            points,
            point_dim,
            offset,
        }
    }
    pub fn npoints(&self) -> Index {
        (self.points.len() / self.point_dim as usize) as Index
    }
}

impl SequenceTransformation for UniformPoints {
    #[inline]
    fn delta_dim(&self) -> Dim {
        self.point_dim
    }
    #[inline]
    fn len(&self, parent_len: Index) -> Index {
        parent_len * (self.points.len() as Index / self.point_dim as Index)
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        self.offset += amount;
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        self.offset -= amount;
    }
    fn apply_inplace(&self, index: Index, coordinate: &mut [f64]) -> Index {
        let point_dim = self.point_dim as usize;
        let coordinate = &mut coordinate[self.offset as usize..];
        coordinate.copy_within(..coordinate.len() - point_dim, point_dim);
        let npoints = (self.points.len() / point_dim) as Index;
        let ipoint = index % npoints;
        let offset = ipoint as usize * point_dim;
        let points: &[f64] =
            unsafe { std::mem::transmute(&self.points[offset..offset + point_dim]) };
        coordinate[..point_dim].copy_from_slice(points);
        index / npoints
    }
}

impl DescribeOperator for UniformPoints {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Coordinate(self.offset, self.point_dim, 0, self.npoints())
    }
}

/// An operator that maps a sequence of elements to another sequence of elements.
///
/// Given a sequence of elements an [`Operator`] defines a new sequence. For
/// example [`Operator::Children`] gives the sequence of child elements and
/// [`Operator::Take`] gives a subset of the input sequence.
///
/// All variants of [`Operator`] apply some operation to either every element of
/// the parent sequence, variants [`Operator::Children`], [`Operator::Edges`]
/// and [`Operator::UniformPoints`], or to consecutive chunks of the input
/// sequence, in which case the size of the chunks is included in the variant
/// and the input sequence is assumed to be a multiple of the chunk size long.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operator {
    /// The transpose of a sequence: the input sequence is reshaped to `(_,
    /// len1, len2)`, the last two axes are swapped and the result is
    /// flattened.
    Transpose(Transpose),
    /// A subset of a sequence: the input sequence is reshaped to `(_, len)`,
    /// the given `indices` are taken from the last axis and the result is
    /// flattened.
    Take(Take),
    /// The children of a every element of a sequence.
    Children(Children),
    /// The edges of a every element of a sequence.
    Edges(Edges),
    UniformPoints(UniformPoints),
}

macro_rules! impl_from_for_operator {
    ($($Variant:ident),*) => {$(
        impl From<$Variant> for Operator {
            fn from(variant: $Variant) -> Self {
                Self::$Variant(variant)
            }
        }
    )*}
}

impl_from_for_operator! {Transpose, Take, Children, Edges, UniformPoints}

impl Operator {
    /// Construct a new operator that transposes a sequence of elements.
    pub fn new_transpose(len1: Index, len2: Index) -> Self {
        Transpose::new(len1, len2).into()
    }
    /// Construct a new operator that takes a subset of a sequence of elements.
    pub fn new_take(indices: impl Into<Box<[Index]>>, len: Index) -> Self {
        Take::new(indices, len).into()
    }
    /// Construct a new operator that maps a sequence of elements to its children.
    pub fn new_children(simplex: Simplex, offset: Dim) -> Self {
        Children::new(simplex, offset).into()
    }
    /// Construct a new operator that maps a sequence of elements to its edges.
    pub fn new_edges(simplex: Simplex, offset: Dim) -> Self {
        Edges::new(simplex, offset).into()
    }
    /// Construct a new operator that adds points to every element of a sequence.
    pub fn new_uniform_points(points: Box<[f64]>, point_dim: Dim, offset: Dim) -> Self {
        UniformPoints::new(points, point_dim, offset).into()
    }
    pub fn swap(&self, other: &Self) -> Option<Vec<Self>> {
        let mut other = other.clone();
        swap(self, &mut other).map(|tail| iter::once(other).chain(tail.into_iter()).collect())
    }
}

macro_rules! dispatch {
    ($vis:vis fn $fn:ident(&$self:ident $(, $arg:ident: $ty:ty)*) $($ret:tt)*) => {
        #[inline]
        $vis fn $fn(&$self $(, $arg: $ty)*) $($ret)* {
            match $self {
                Operator::Transpose(var) => var.$fn($($arg),*),
                Operator::Take(var) => var.$fn($($arg),*),
                Operator::Children(var) => var.$fn($($arg),*),
                Operator::Edges(var) => var.$fn($($arg),*),
                Operator::UniformPoints(var) => var.$fn($($arg),*),
            }
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $($ret:tt)*) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $($ret)* {
            match $self {
                Operator::Transpose(var) => var.$fn($($arg),*),
                Operator::Take(var) => var.$fn($($arg),*),
                Operator::Children(var) => var.$fn($($arg),*),
                Operator::Edges(var) => var.$fn($($arg),*),
                Operator::UniformPoints(var) => var.$fn($($arg),*),
            }
        }
    };
}

impl SequenceTransformation for Operator {
    dispatch! {fn delta_dim(&self) -> Dim}
    dispatch! {fn len(&self, parent_len: Index) -> Index}
    dispatch! {fn increment_offset(&mut self, amount: Dim)}
    dispatch! {fn decrement_offset(&mut self, amount: Dim)}
    dispatch! {fn apply_inplace(&self, index: Index, coordinate: &mut [f64]) -> Index}
    dispatch! {fn apply_many_inplace(&self, index: Index, coordinates: &mut [f64], dim: Dim) -> Index}
    dispatch! {fn apply(&self, index: Index, coordinate: &[f64]) -> (Index, Vec<f64>)}
    dispatch! {fn apply_many(&self, index: Index, coordinates: &[f64], dim: Dim) -> (Index, Vec<f64>)}
}

impl DescribeOperator for Operator {
    dispatch! {fn operator_kind(&self) -> OperatorKind}
    dispatch! {fn as_children(&self) -> Option<&Children>}
    dispatch! {fn as_children_mut(&mut self) -> Option<&mut Children>}
    dispatch! {fn as_edges(&self) -> Option<&Edges>}
    dispatch! {fn as_edges_mut(&mut self) -> Option<&mut Edges>}
}

impl std::fmt::Debug for Operator {
    dispatch! {fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result}
}

fn swap<L, R>(l: &L, r: &mut R) -> Option<Vec<Operator>>
where
    L: DescribeOperator + SequenceTransformation + Into<Operator>,
    R: DescribeOperator + SequenceTransformation,
{
    if let (Some(edges), Some(children)) = (l.as_edges(), r.as_children_mut()) {
        if edges.offset == children.offset && edges.simplex.edge_dim() == children.simplex.dim() {
            let simplex = edges.simplex;
            let indices = simplex.swap_edges_children_map();
            let take = Operator::new_take(indices, simplex.nchildren() * simplex.nedges());
            children.simplex = simplex;
            return Some(vec![l.clone().into(), take]);
        }
    }
    use OperatorKind::*;
    match (l.operator_kind(), r.operator_kind()) {
        (Index(l_nout, l_nin), Coordinate(_, _, _, r_gen)) => Some(vec![
            Operator::new_transpose(r_gen, l_nout),
            l.clone().into(),
            Operator::new_transpose(l_nin, r_gen),
        ]),
        (Coordinate(l_off, _, l_nin, l_gen), Coordinate(r_off, r_nout, _, r_gen)) => {
            if l_off + l_nin <= r_off {
                r.increment_offset(l.delta_dim());
                Some(vec![
                    l.clone().into(),
                    Operator::new_transpose(l_gen, r_gen),
                ])
            } else if l_off >= r_off + r_nout {
                let mut l = l.clone();
                l.decrement_offset(r.delta_dim());
                Some(vec![l.into(), Operator::new_transpose(l_gen, r_gen)])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A chain of [`Operator`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    rev_operators: Vec<Operator>,
}

impl Chain {
    #[inline]
    pub fn new<Operators>(operators: Operators) -> Self
    where
        Operators: IntoIterator<Item = Operator>,
        Operators::IntoIter: DoubleEndedIterator,
    {
        Chain {
            rev_operators: operators.into_iter().rev().collect(),
        }
    }
    /// Returns a clone of this [`Chain`] with the given `operator` appended.
    #[inline]
    pub fn clone_and_push(&self, operator: Operator) -> Self {
        Self::new(
            self.rev_operators
                .iter()
                .rev()
                .cloned()
                .chain(iter::once(operator)),
        )
    }
    #[inline]
    pub fn iter_operators(&self) -> impl Iterator<Item = &Operator> + DoubleEndedIterator {
        self.rev_operators.iter().rev()
    }
    fn split_heads(&self) -> BTreeMap<Operator, Vec<Operator>> {
        let mut heads = BTreeMap::new();
        'a: for (i, head) in self.rev_operators.iter().enumerate() {
            let mut rev_tail: Vec<_> = self.rev_operators.iter().take(i).cloned().collect();
            let mut head = head.clone();
            for op in self.rev_operators.iter().skip(i + 1) {
                if let Some(ops) = swap(op, &mut head) {
                    rev_tail.extend(ops.into_iter().rev());
                } else {
                    continue 'a;
                }
            }
            heads.insert(head, rev_tail);
        }
        'b: for (i, op) in self.rev_operators.iter().enumerate() {
            if let Operator::Edges(Edges {
                simplex: Simplex::Line,
                offset,
            }) = op
            {
                let simplex = Simplex::Line;
                let mut rev_tail: Vec<_> = self.rev_operators.iter().take(i).cloned().collect();
                let mut head = Operator::new_children(simplex, *offset);
                let indices = simplex.swap_edges_children_map();
                let take = Operator::new_take(indices, simplex.nchildren() * simplex.nedges());
                rev_tail.push(take);
                rev_tail.push(op.clone());
                for op in self.rev_operators.iter().skip(i + 1) {
                    if let Some(ops) = swap(op, &mut head) {
                        rev_tail.extend(ops.into_iter().rev());
                    } else {
                        continue 'b;
                    }
                }
                heads.insert(head, rev_tail);
            }
        }
        heads
    }
    /// Remove and return the common prefix of two chains, transforming either if necessary.
    pub fn remove_common_prefix(&self, other: &Self) -> (Self, Self, Self) {
        let mut common = Vec::new();
        let mut rev_a = self.rev_operators.clone();
        let mut rev_b = other.rev_operators.clone();
        let mut i = 0;
        while !rev_a.is_empty() && !rev_b.is_empty() {
            i += 1;
            if i > 10 {
                break;
            }
            if rev_a.last() == rev_b.last() {
                common.push(rev_a.pop().unwrap());
                rev_b.pop();
                continue;
            }
            let heads_a = Chain::new(rev_a.iter().rev().cloned()).split_heads();
            let heads_b = Chain::new(rev_b.iter().rev().cloned()).split_heads();
            let candidates: Vec<_> = heads_a.iter().filter_map(|(h, a)| heads_b.get(h).map(|b| (h, a, b))).collect();
            if let Some((head, a, b)) = candidates.into_iter().min_by_key(|(_, a, b)| std::cmp::max(a.len(), b.len())) {
                common.push(head.clone());
                rev_a = a.clone();
                rev_b = b.clone();
                continue;
            }
            break;
        }
        let common = if rev_a.is_empty()
            && (!rev_b.is_empty() || self.rev_operators.len() <= other.rev_operators.len())
        {
            self.clone()
        } else if rev_b.is_empty() {
            other.clone()
        } else {
            Self::new(common)
        };
        (
            common,
            Self::new(rev_a.into_iter().rev()),
            Self::new(rev_b.into_iter().rev()),
        )
    }
}

impl SequenceTransformation for Chain {
    #[inline]
    fn delta_dim(&self) -> Dim {
        self.rev_operators.iter().map(|op| op.delta_dim()).sum()
    }
    #[inline]
    fn len(&self, parent_len: Index) -> Index {
        self.rev_operators
            .iter()
            .rfold(parent_len, |len, op| op.len(len))
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        for op in self.rev_operators.iter_mut() {
            op.increment_offset(amount);
        }
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        for op in self.rev_operators.iter_mut() {
            op.decrement_offset(amount);
        }
    }
    #[inline]
    fn apply_inplace(&self, index: Index, coordinate: &mut [f64]) -> Index {
        self.rev_operators
            .iter()
            .fold(index, |index, op| op.apply_inplace(index, coordinate))
    }
    #[inline]
    fn apply_many_inplace(&self, index: Index, coordinates: &mut [f64], dim: Dim) -> Index {
        self.rev_operators.iter().fold(index, |index, op| {
            op.apply_many_inplace(index, coordinates, dim)
        })
    }
}

//#[derive(Debug, Clone)]
//pub struct ConcatChain(Vec<(Chain, Index)>);

#[derive(Debug, Clone)]
pub struct Topology {
    transforms: Chain,
    dim: Dim,
    root_len: Index,
    len: Index,
}

impl Topology {
    pub fn new(dim: Dim, len: Index) -> Self {
        Self {
            transforms: Chain::new([]),
            dim,
            root_len: len,
            len,
        }
    }
    pub fn derive(&self, operator: Operator) -> Self {
        Self {
            root_len: self.root_len,
            len: operator.len(self.len),
            dim: self.dim - operator.delta_dim(),
            transforms: self.transforms.clone_and_push(operator),
        }
    }
}

impl Mul for &Topology {
    type Output = Topology;

    fn mul(self, other: &Topology) -> Topology {
        Topology {
            transforms: Chain::new(
                iter::once(Operator::new_transpose(other.root_len, self.root_len))
                    .chain(self.transforms.iter_operators().cloned())
                    .chain(iter::once(Operator::new_transpose(
                        self.len,
                        other.root_len,
                    )))
                    .chain(other.transforms.iter_operators().map(|op| {
                        let mut op = op.clone();
                        op.increment_offset(self.dim);
                        op
                    })),
            ),
            dim: self.dim + other.dim,
            root_len: self.root_len * other.root_len,
            len: self.len * other.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use Simplex::*;

    macro_rules! assert_eq_op_apply {
        ($op:expr, $ii:expr, $ic:expr, $oi:expr, $oc:expr) => {{
            let ic = $ic;
            let oc = $oc;
            let mut work = oc.clone();
            for i in 0..ic.len() {
                work[i] = ic[i];
            }
            for i in ic.len()..oc.len() {
                work[i] = 0.0;
            }
            assert_eq!($op.apply_inplace($ii, &mut work), $oi);
            assert_abs_diff_eq!(work[..], oc[..]);
        }};
    }

    #[test]
    fn apply_children_line() {
        let op = Operator::new_children(Line, 0);
        assert_eq_op_apply!(op, 0 * 2 + 0, [0.0], 0, [0.0]);
        assert_eq_op_apply!(op, 1 * 2 + 0, [1.0], 1, [0.5]);
        assert_eq_op_apply!(op, 2 * 2 + 1, [0.0], 2, [0.5]);
        assert_eq_op_apply!(op, 3 * 2 + 1, [1.0], 3, [1.0]);
        assert_eq_op_apply!(op, 0, [0.0, 2.0], 0, [0.0, 2.0]);
        assert_eq_op_apply!(op, 1, [0.0, 3.0, 4.0], 0, [0.5, 3.0, 4.0]);
        let op = Operator::new_children(Line, 1);
        assert_eq_op_apply!(op, 1, [2.0, 0.0], 0, [2.0, 0.5]);
        assert_eq_op_apply!(op, 1, [3.0, 0.0, 4.0], 0, [3.0, 0.5, 4.0]);
    }

    #[test]
    fn apply_edges_line() {
        let op = Operator::new_edges(Line, 0);
        assert_eq_op_apply!(op, 0, [], 0, [1.0]);
        assert_eq_op_apply!(op, 3, [], 1, [0.0]);
        assert_eq_op_apply!(op, 4, [], 2, [1.0]);
        assert_eq_op_apply!(op, 7, [], 3, [0.0]);
        assert_eq_op_apply!(op, 0, [2.0], 0, [1.0, 2.0]);
        assert_eq_op_apply!(op, 1, [3.0, 4.0], 0, [0.0, 3.0, 4.0]);
        let op = Operator::new_edges(Line, 1);
        assert_eq_op_apply!(op, 0, [2.0], 0, [2.0, 1.0]);
        assert_eq_op_apply!(op, 0, [3.0, 4.0], 0, [3.0, 1.0, 4.0]);
    }

    // #[test]
    // fn apply_edges_square() {
    //     let op = Operator::Edges {
    //         simplices: Box::new([Line, Line]),
    //         offset: 0,
    //     };
    //     assert_eq!(op.apply(0 * 4 + 0, &[0.0]), (0, vec![1.0, 0.0]));
    //     assert_eq!(op.apply(1 * 4 + 0, &[1.0]), (1, vec![1.0, 1.0]));
    //     assert_eq!(op.apply(2 * 4 + 1, &[0.0]), (2, vec![0.0, 0.0]));
    //     assert_eq!(op.apply(3 * 4 + 1, &[1.0]), (3, vec![0.0, 1.0]));
    //     assert_eq!(op.apply(4 * 4 + 2, &[0.0]), (4, vec![0.0, 1.0]));
    //     assert_eq!(op.apply(5 * 4 + 2, &[1.0]), (5, vec![1.0, 1.0]));
    //     assert_eq!(op.apply(6 * 4 + 3, &[0.0]), (6, vec![0.0, 0.0]));
    //     assert_eq!(op.apply(7 * 4 + 3, &[1.0]), (7, vec![1.0, 0.0]));
    //     assert_eq!(op.apply(0, &[0.0, 2.0]), (0, vec![1.0, 0.0, 2.0]));
    //     assert_eq!(op.apply(1, &[0.0, 3.0, 4.0]), (0, vec![0.0, 0.0, 3.0, 4.0]));
    // }

    #[test]
    fn apply_transpose_index() {
        let op = Operator::new_transpose(2, 3);
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..3 {
                    assert_eq!(
                        op.apply((i * 2 + j) * 3 + k, &[]),
                        ((i * 3 + k) * 2 + j, vec![])
                    );
                }
            }
        }
    }

    #[test]
    fn apply_take_all() {
        let op = Operator::new_take([3, 2, 0, 4, 1], 5); // inverse: [2, 4, 1, 0, 3]
        assert_eq_op_apply!(op, 0, [], 3, []);
        assert_eq_op_apply!(op, 6, [1.0], 7, [1.0]);
        assert_eq_op_apply!(op, 12, [2.0, 3.0], 10, [2.0, 3.0]);
        assert_eq_op_apply!(op, 18, [], 19, []);
        assert_eq_op_apply!(op, 24, [], 21, []);
    }

    #[test]
    fn apply_take_some() {
        let op = Operator::new_take([4, 0, 1], 5); // inverse: [1, 2, x, x, 0]
        assert_eq_op_apply!(op, 0, [], 4, []);
        assert_eq_op_apply!(op, 4, [1.0], 5, [1.0]);
        assert_eq_op_apply!(op, 8, [2.0, 3.0], 11, [2.0, 3.0]);
    }

    #[test]
    fn apply_uniform_points() {
        let op = Operator::new_uniform_points(Box::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), 2, 0);
        assert_eq_op_apply!(op, 0, [], 0, [0.0, 1.0]);
        assert_eq_op_apply!(op, 4, [6.0], 1, [2.0, 3.0, 6.0]);
        assert_eq_op_apply!(op, 8, [7.0, 8.0], 2, [4.0, 5.0, 7.0, 8.0]);
    }

    #[test]
    fn mul_topo() {
        let xtopo = Topology::new(1, 2).derive(Operator::new_children(Line, 0));
        let ytopo = Topology::new(1, 3).derive(Operator::new_children(Line, 0));
        let xytopo = &xtopo * &ytopo;
        assert_eq!(xtopo.len, 4);
        assert_eq!(ytopo.len, 6);
        assert_eq!(xytopo.len, 24);
        assert_eq!(xytopo.root_len, 6);
        for i in 0..4 {
            for j in 0..6 {
                let x = xtopo.transforms.apply_many(i, &[0.0, 0.0, 1.0, 1.0], 1).1;
                let y = ytopo.transforms.apply_many(j, &[0.0, 1.0, 0.0, 1.0], 1).1;
                let mut xy = Vec::with_capacity(8);
                for k in 0..4 {
                    xy.push(x[k]);
                    xy.push(y[k]);
                }
                assert_eq!(
                    xytopo.transforms.apply_many(
                        i * 6 + j,
                        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                        2
                    ),
                    ((i / 2) * 3 + j / 2, xy),
                );
            }
        }
    }

    macro_rules! assert_equiv_topo {
        ($topo1:expr, $topo2:expr$(, $simplex:ident)*) => {
            #[allow(unused_mut)]
            let mut topo1 = $topo1.clone();
            #[allow(unused_mut)]
            let mut topo2 = $topo2.clone();
            assert_eq!(topo1.dim, topo2.dim, "topos have different dim");
            assert_eq!(topo1.len, topo2.len, "topos have different len");
            assert_eq!(topo1.root_len, topo2.root_len, "topos have different root_len");
            let from_dim = 0$(+$simplex.dim())*;
            assert_eq!(topo1.dim, from_dim, "dimension of topo differs from dimension of given simplices");
            let nelems = topo1.len;
            $(
                let points = Operator::new_uniform_points(
                    $simplex.vertices().into(),
                    $simplex.dim(),
                    0,
                );
                topo1 = topo1.derive(points.clone());
                topo2 = topo2.derive(points.clone());
            )*
            let npoints = topo1.len;
            let mut coord1: Vec<_> = iter::repeat(0.0).take((topo1.dim + topo1.transforms.delta_dim()) as usize).collect();
            let mut coord2 = coord1.clone();
            for i in 0..topo1.len {
                let ielem = i / (npoints / nelems);
                assert_eq!(
                    topo1.transforms.apply_inplace(i, &mut coord1),
                    topo2.transforms.apply_inplace(i, &mut coord2),
                    "topo1 and topo2 map element {ielem} to different root elements"
                );
                assert_abs_diff_eq!(coord1[..], coord2[..]);
            }
        };
    }

    #[test]
    fn swap_edges_children_1d() {
        let topo1 = Topology::new(1, 3).derive(Operator::new_edges(Line, 0));
        let topo2 = Topology::new(1, 3)
            .derive(Operator::new_children(Line, 0))
            .derive(Operator::new_edges(Line, 0))
            .derive(Operator::new_take([2, 1], 4));
        assert_equiv_topo!(topo1, topo2);
    }

    #[test]
    fn swap_take_children() {
        let take = Operator::new_take([2, 3, 1], 5);
        let children = Operator::new_children(Line, 0);
        let swapped = vec![
            children.clone(),
            Operator::new_transpose(2, 5),
            take.clone(),
            Operator::new_transpose(3, 2),
        ];
        let base = Topology::new(1, 5);
        assert_eq!(take.swap(&children), Some(swapped.clone()));
        assert_equiv_topo!(
            base.derive(take).derive(children),
            swapped
                .iter()
                .cloned()
                .fold(base.clone(), |t, o| t.derive(o)),
            Line
        );
    }

    #[test]
    fn swap_take_edges() {
        let take = Operator::new_take([2, 3, 1], 5);
        let edges = Operator::new_edges(Line, 0);
        let swapped = vec![
            edges.clone(),
            Operator::new_transpose(2, 5),
            take.clone(),
            Operator::new_transpose(3, 2),
        ];
        let base = Topology::new(1, 5);
        assert_eq!(take.swap(&edges), Some(swapped.clone()));
        assert_equiv_topo!(
            base.derive(take).derive(edges),
            swapped
                .iter()
                .cloned()
                .fold(base.clone(), |t, o| t.derive(o))
        );
    }

    macro_rules! fn_test_operator_swap {
        ($name:ident, $len:expr $(, $simplex:ident)*; $op1:expr, $op2:expr,) => {
            #[test]
            fn $name() {
                let op1: Operator = $op1;
                let op2: Operator = $op2;
                let swapped = op1.swap(&op2).expect("not swapped");
                println!("op1: {op1:?}");
                println!("op2: {op2:?}");
                println!("swapped: {swapped:?}");
                let root_dim = op1.delta_dim() + op2.delta_dim() $(+ $simplex.dim())*;
                let base = Topology::new(root_dim, 1);
                let topo1 = [op1, op2].iter().fold(base.clone(), |t, o| t.derive(o.clone()));
                let topo2 = swapped.iter().fold(base, |t, o| t.derive(o.clone()));
                let len = $len;
                assert_eq!(topo1.len, len, "unswapped topo has unexpected length");
                assert_eq!(topo2.len, len, "swapped topo has unexpected length");
                assert_equiv_topo!(topo1, topo2 $(, $simplex)*);
            }
        }
    }

    fn_test_operator_swap! {
        swap_edges_children_triangle1, 6, Line, Line;
        Operator::new_edges(Triangle, 0),
        Operator::new_children(Line, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_lt_children, 8, Triangle, Line;
        Operator::new_children(Triangle, 0),
        Operator::new_children(Line, 2),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_gt_children, 8, Line, Triangle;
        Operator::new_children(Line, 2),
        Operator::new_children(Triangle, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_lt_children, 6, Line, Line;
        Operator::new_edges(Triangle, 0),
        Operator::new_children(Line, 1),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_gt_children, 6, Line, Line;
        Operator::new_edges(Triangle, 1),
        Operator::new_children(Line, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_lt_edges, 6, Line, Line;
        Operator::new_children(Line, 0),
        Operator::new_edges(Triangle, 1),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_gt_edges, 6, Line, Line;
        Operator::new_children(Line, 2),
        Operator::new_edges(Triangle, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_lt_edges, 6, Line;
        Operator::new_edges(Line, 0),
        Operator::new_edges(Triangle, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_gt_edges, 6, Line;
        Operator::new_edges(Line, 2),
        Operator::new_edges(Triangle, 0),
    }

    #[test]
    fn split_heads() {
        let chain = Chain::new([
            Operator::new_edges(Triangle, 1),
            Operator::new_children(Line, 0),
            Operator::new_edges(Line, 2),
            Operator::new_children(Line, 1),
            Operator::new_children(Line, 0),
        ]);
        let desired = chain
            .iter_operators()
            .cloned()
            .fold(Topology::new(4, 1), |topo, op| topo.derive(op));
        for (head, tail) in chain.split_heads().into_iter() {
            let actual = iter::once(head)
                .chain(tail.into_iter().rev())
                .fold(Topology::new(4, 1), |topo, op| topo.derive(op));
            assert_equiv_topo!(actual, desired, Line, Line);
        }
    }

    #[test]
    fn remove_common_prefix() {
        let a = Chain::new([Operator::new_children(Line, 0), Operator::new_children(Line, 0)]);
        let b = Chain::new([Operator::new_edges(Line, 0)]);
        assert_eq!(
            a.remove_common_prefix(&b),
            (
                Chain::new([Operator::new_children(Line, 0), Operator::new_children(Line, 0)]),
                Chain::new([]),
                Chain::new([Operator::new_edges(Line, 0), Operator::new_take([2, 1], 4), Operator::new_take([2, 1], 4)]),
            )
        );
    }
}
