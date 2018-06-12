use std::boxed::Box;
use std::sync::Arc;
use std::marker::PhantomData;
use std::iter::*;



struct Fun<A, B> {
    function: Box<Fn(A) -> B>,
}

impl<A, B> Fun<A, B> {
    fn apply(&self) -> &(Fn(A) -> B) {
        self.function.as_ref()
    }
}

trait Function: Sized {
    fn function<B, F>(fun: F) -> PFn2<Self, B>
    where F: Fn(Self) -> B;
}

// functionMap :: Function b => (a -> b) -> (b -> a) -> (a -> c) -> a :-> c 

fn function_map<A, B, C, F1, F2, F3>(_f1: F1, _f2: F2, _f3: F3) -> PFn2<A, C>
where
    B: Function,
    F1: Fn(A) -> B,
    F2: Fn(B) -> A,
    F3: Fn(A) -> C,
{
    unimplemented!()
}

struct PFn2<A, B> {
    __phantom: PhantomData<(A, B)>
}





/*
/// data a :-> c where
///   Pair  :: (a :-> (b :-> c)) -> ((a,b) :-> c)
///   (:+:) :: (a :-> c) -> (b :-> c) -> (Either a b :-> c)
///   Unit  :: c -> (() :-> c)
///   Nil   :: a :-> c
///   Table :: Eq a => [(a,c)] -> (a :-> c)
///   Map   :: (a -> b) -> (b -> a) -> (b :-> c) -> (a :-> c)
*/


pub trait PFn<X, Y> {
    type Table: Iterator<Item = (X, Y)>;

    fn table(&self) -> Self::Table;

    fn apply(&self, x: X) -> Option<Y>;

    //fn shrink(&self, y_shrink: YShrinker<Y>) -> PFnShrinker<X, Y>;
}

// type YShrinker<'a, Y> = &'a (Fn(Y) -> Box<Iterator<Item = Y>>);
// type PFnShrinker<X, Y> = Box<Iterator<Item = BoxedPFn<X, Y>>>;

//==============================================================================
// Boxing a PFn:
//==============================================================================

/// A boxed PFn where the concrete PFn has been erased. This let's us talk
/// about `PFn` as a type.
pub struct BoxedPFn<X, Y> {
    /// The trait object we are delegating to.
    pfn: Box<PFn<X, Y, Table = Box<Iterator<Item = (X, Y)>>>>
}

impl<X, Y> PFn<X, Y> for BoxedPFn<X, Y>
where
    X: 'static, Y: 'static,
{
    type Table = Box<Iterator<Item = (X, Y)>>;

    fn table(&self) -> Self::Table { self.pfn.table() }

    fn apply(&self, args: X) -> Option<Y> { self.pfn.apply(args) }

    /*
    fn shrink(&self, y_shrink: YShrinker<Y>) -> PFnShrinker<X, Y> {
        self.pfn.shrink(y_shrink)
    }
    */
}

/// A `PFn` that boxes the table. This is an implementation detail of `BoxedPFn`.
struct BoxedWrapper<F> {
    /// The function we are wrapping.
    pfn: F
}

impl<X, Y, F> PFn<X, Y> for BoxedWrapper<F>
where
    X: 'static, Y: 'static, <F as PFn<X, Y>>::Table: 'static,
    F: PFn<X, Y>,
{
    type Table = Box<Iterator<Item = (X, Y)>>;
    fn table(&self) -> Self::Table { Box::new(self.pfn.table()) }

    fn apply(&self, args: X) -> Option<Y> { self.pfn.apply(args) }

    /*
    fn shrink(&self, y_shrink: YShrinker<Y>) -> PFnShrinker<X, Y> {
        Box::new(self.pfn.shrink(y_shrink).map(|pfn| pfn.boxed()))
    }
    */
}




//------------------------------------------------------------------------------
// [Holy Trinity]
//
// Using unit, sum, and pair we can define PFn for most types.
//------------------------------------------------------------------------------

//==============================================================================
// [Holy Trinity] The `Unit` variant:
//==============================================================================

/// A `PFn<(), Y>` that produces `Y` using `F: Fn() -> Y`.
pub fn unit<Y: Clone>(output: Y) -> UnitPFn<Y> { UnitPFn { output } }

/// See `unit`.
pub struct UnitPFn<Y> {
    /// The result `Y`
    output: Y
}

impl<Y: 'static + Clone> PFn<(), Y> for UnitPFn<Y> {
    type Table = Once<((), Y)>;

    fn table(&self) -> Self::Table { once( ( (), self.output.clone() ) ) }

    fn apply(&self, _: ()) -> Option<Y> { Some( self.output.clone() ) }

    /*
    fn shrink(&self, y_shrink: YShrinker<Y>) -> PFnShrinker<(), Y> {
        Box::new(
            once(nil().boxed()).chain(
                y_shrink(self.output.clone()).map(|y| unit(y).boxed())
            )
        )
    }
    */
}

//==============================================================================
// [Holy Trinity] The `:+:` variant:
//==============================================================================

/// A `PFn<Either<X, Y>, Z>` made up of a `PFn<X, Z>` and `PFn<Y, Z>`.
/// The `table` is the union of the tables of two functions while function
/// application simply uses the left 
pub fn sum<X, Y, Z, F, G>(left: F, right: G) -> SumPFn<F, G>
where
     F: PFn<X, Z>,
     G: PFn<Y, Z>,
{
    SumPFn { left, right }
}

/// See `sum`.
pub struct SumPFn<F, G> {
    /// The left function PFn<X, Z>.
    left: F,
    /// The right function PFn<Y, Z>.
    right: G
}

enum Either<A, B> { Left(A), Right(B), }

impl<X, Y, Z, F, G> PFn<Either<X, Y>, Z> for SumPFn<F, G>
where
    F: PFn<X, Z>,
    G: PFn<Y, Z>
{
    type Table = Chain<
        Map<F::Table, fn((X, Z)) -> (Either<X, Y>, Z)>,
        Map<G::Table, fn((Y, Z)) -> (Either<X, Y>, Z)>,
    >;

    fn table(&self) -> Self::Table {
        let l: fn((X, Z)) -> (Either<X, Y>, Z) = |(x, z)| (Either::Left(x), z);
        let r: fn((Y, Z)) -> (Either<X, Y>, Z) = |(y, z)| (Either::Right(y), z);
        self.left.table().map(l).chain(self.right.table().map(r))
    }

    fn apply(&self, exy: Either<X, Y>) -> Option<Z> {
        match exy {
            Either::Left(x) => self.left.apply(x),
            Either::Right(y) => self.right.apply(y),
        }
    }
}


//==============================================================================
// [Holy Trinity] The `Pair` variant:
//==============================================================================

/// A `PFn<(X, Y), Z>`, that is a function from a pair `(X, Y)` to `Z`.
/// From this, we can then construct functions taking arbitrary N-tuples.
pub fn pair<X, Y, Z, F, G> (pfn: F) -> PairPFn<X, F, G>
where
    X: Clone + 'static,
    Y: 'static,
    Z: 'static,
    F: PFn<X, G>,
    G: PFn<Y, Z> + 'static
{
    PairPFn { pfn, phantom: PhantomData }
}

#[derive(Clone)]
pub struct PairPFn<X, F, G> {
    /// The outer function X -> PFn<Y, Z>.
    pfn: F,
    /// Make the compiler happy.
    phantom: PhantomData<(X, G)>
}

impl<X, Y, Z, F, G> PFn<(X, Y), Z> for PairPFn<X, F, G>
where
    X: Clone + 'static,
    Y: 'static,
    Z: 'static,
    F: PFn<X, G>,
    G: PFn<Y, Z> + 'static,
{
    type Table = FlatMap<
        <F as PFn<X, G>>::Table,
        PairPFnInner<X, <G as PFn<Y, Z>>::Table>,
        fn((X, G)) -> PairPFnInner<X, <G as PFn<Y, Z>>::Table>
    >;
    fn table(&self) -> Self::Table {
        self.pfn.table().flat_map(
            |(fst, snd_pfn)| PairPFnInner { fst, snd_table: snd_pfn.table() })
    }

    fn apply(&self, (x, y): (X, Y)) -> Option<Z> {
        self.pfn.apply(x).and_then(|snd_pfn| snd_pfn.apply(y))
    }

    /*
    fn shrink(&self, z_shrink: YShrinker<Z>) -> PFnShrinker<(X, Y), Z> {
        let y_shrink = move |q| {
            let a = q.shrink(z_shrink);
            a
        };
        self.shrink(&y_shrink)
    }
    */
}

/// The inner iterator used in `.flat_map(..)` for the table of `pair`
/// (implementation detail).
pub struct PairPFnInner<X, I> {
    /// A specific X in the `fst pair` of a pair.
    fst: X,
    /// The table of `snd`s in `snd pair`. The `snd` is dependent on `fst`.
    snd_table: I
}

impl<X: Clone, Y, Z, I: Iterator<Item = (Y, Z)>>
Iterator for PairPFnInner<X, I> {
    type Item = ((X, Y), Z);
    fn next(&mut self) -> Option<Self::Item> {
        let xclone = self.fst.clone();
        self.snd_table.next().map(move |(y, z)| ((xclone, y), z) )
    }
}


//==============================================================================
// The `Nil` variant:
//==============================================================================

/// A `PFn<X, Y>` that never returns an `Y` for any `X`. This is the main
/// source of partiality other than `table(empty())`.
pub fn nil<X, Y>() -> NilPFn<X, Y> { NilPFn(PhantomData) }

/// See `nil`.
pub struct NilPFn<X, Y>(PhantomData<(X, Y)>);

impl<X, Y> PFn<X, Y> for NilPFn<X, Y> {
    type Table = Empty<(X, Y)>;

    fn table(&self) -> Self::Table { empty() }

    fn apply(&self, _: X) -> Option<Y> { None }
}

//==============================================================================
// The `Table` variant:
//==============================================================================

/// A `PFn<X, Y>` defined by a function table of `(X, Y)` bindings.
///
/// We define function application by a simple linear search in `X`
/// for equality and return the first `Y` where the `X` that matches.
///
/// We use a cloneable iterator instead of a `Vec` so that we can be
/// lazy and not have to keep the whole table in memory at the same time.
pub fn table<X, Y, I>(table: I) -> TablePFn<I>
where
    X: PartialEq,
    I: Iterator<Item = (X, Y)> + Clone,
{
    TablePFn { table }
}

/// See `table`.
pub struct TablePFn<I> {
    /// The function table that defines this `PFn<X, Y>`.
    table: I
}

impl<X, Y, I> PFn<X, Y> for TablePFn<I>
where
    X: PartialEq,
    I: Iterator<Item = (X, Y)> + Clone,
{
    type Table = I;

    fn table(&self) -> Self::Table { self.table.clone() }

    fn apply(&self, x: X) -> Option<Y> {
        self.table.clone().find(|&(ref xp, _)| xp == &x).map(|(_, c)| c)
    }
}

//==============================================================================
// The `Map` variant:
//==============================================================================

/// We `Map` a `PFn<Y, Z>` into a `PFn<X, Z>` using a bijection / isomorphism
/// between `X` and `Y`. This bijection is defined by `to : X -> Y` and
/// `from : Y -> X`. We assume that `forall x. x ≡ from(to(x))` or 
/// equivalently: `from . to ≡ id`.
pub fn map<X, Y, Z, F, G, H>(to: F, from: G, pfn: H) -> MapPFn<F, G, H>
where
    X: 'static, Y: 'static, Z: 'static,
    F: Fn(X) -> Y,
    G: Fn(Y) -> X,
    H: PFn<Y, Z>,
{
    MapPFn { to: Arc::new(to), from: Arc::new(from), pfn }
}

/// See `map`.
pub struct MapPFn<F, G, H> {
    /// The `to : X -> Y` part of a bijection.
    to: Arc<F>,
    /// The `from : Y -> X` part of a bijection.
    from: Arc<G>,
    /// The partial function `PFn<Y, Z>` we are mapping into `PFn<X, Z>`.
    pfn: H
}

impl<X, Y, Z, F, G, H> PFn<X, Z> for MapPFn<F, G, H>
where
    X: 'static, Y: 'static, Z: 'static, F: 'static, G: 'static,
    F: Fn(X) -> Y,
    G: Fn(Y) -> X,
    H: PFn<Y, Z>
{
    type Table = MapPFnTable<<H as PFn<Y, Z>>::Table, G>;

    fn table(&self) -> Self::Table {
        MapPFnTable { iter: self.pfn.table(), from: self.from.clone() }
    }

    fn apply(&self, x: X) -> Option<Z> {
        self.pfn.apply((self.to)(x))
    }

    /*
    fn shrink(&self, y_shrink: YShrinker<Z>) -> PFnShrinker<X, Z> {
        Box::new(MapPFnShrink {
            shrink: self.pfn.shrink(y_shrink),
            to: self.to.clone(),
            from: self.from.clone(),
        })
    }
    */
}

/*
pub struct MapPFnShrink<I, F, G> {
    shrink: I,
    to: Arc<F>,
    from: Arc<G>
}

impl<X, Y, Z, F, G, I> Iterator for MapPFnShrink<I, F, G>
where
    X: 'static, Y: 'static, Z: 'static, F: 'static, G: 'static,
    F: Fn(X) -> Y,
    G: Fn(Y) -> X,
    I: Iterator<Item = BoxedPFn<Y, Z>>
{
    type Item = BoxedPFn<X, Z>;
    fn next(&mut self) -> Option<Self::Item> {
        self.shrink.next().map(|pfn|
            MapPFn { pfn, to: self.to.clone(), from: self.from.clone() }
                .boxed())
    }
}
*/

/// The table of `MapPFn` (implementation detail).
pub struct MapPFnTable<I, F> { iter: I, from: Arc<F> }

impl<X, Y, Z, F, I> Iterator for MapPFnTable<I, F>
where
    F: Fn(Y) -> X,
    I: Iterator<Item = (Y, Z)>,
{
    type Item = (X, Z);
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(x, z)| ((&*self.from)(x), z))
    }
}
