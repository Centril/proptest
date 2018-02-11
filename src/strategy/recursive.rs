//-
// Copyright 2017 Jason Lingle
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::sync::Arc;

use strategy::traits::*;
use strategy::IndFlatten;
use strategy::statics::{Map, MapFn};
use test_runner::*;

/// A branching `MapFn` that picks `yes` (true) or
/// `no` (false) and clones the `Arc`.
struct BranchFn<T> {
    /// Result on true.
    yes: Arc<T>,
    /// Result on false.
    no: Arc<T>,
}

impl<T> Clone for BranchFn<T> {
    fn clone(&self) -> Self {
        Self { yes: Arc::clone(&self.yes), no: Arc::clone(&self.no) }
    }
}

impl<T: fmt::Debug> MapFn<bool> for BranchFn<T> {
    type Output = Arc<T>;
    fn apply(&self, branch: bool) -> Self::Output {
        Arc::clone(if branch { &self.yes } else { &self.no })
    }
}

/// The branching strategy between recuring or not.
type Branch<S> = IndFlatten<Map<::bool::Weighted, BranchFn<S>>>;

/// The `ValueTree` of `RegArgInner`.

enum RecVT<L: Strategy, R: Strategy>
where R::Value: ValueTree<Value = ValueFor<L>> {
    /// See RecArgInner::Leaf.
    Leaf(L::Value),
    /// See RecArgInner::Recur.
    Recur(R::Value),
    /// See RecArgInner::Branch.
    Branch(Box<<Branch<RecArg<L, R>> as Strategy>::Value>),
}

/// The real implementation of `RecArg<L, R>`.
#[derive(Debug)]
enum RecArgInner<L: Strategy, R: Strategy>
where R::Value: ValueTree<Value = ValueFor<L>> {
    /// Leaf strategy in `leaf.prop_recursive(..)`.
    Leaf(L),
    /// Recursive strategy returned in
    /// `leaf.prop_recursive(.., |arg| <return>)`.
    Recur(R),
    /// For `IndFlatten`.
    Branch(Branch<RecArg<L, R>>),
}

impl<L: Strategy, R: Strategy> ValueTree for RecVT<L, R>
where R::Value: ValueTree<Value = ValueFor<L>> {
    type Value = ValueFor<L>;

    fn current(&self) -> Self::Value {
        match *self {
            RecVT::Leaf(ref vt) => vt.current(),
            RecVT::Recur(ref vt) => vt.current(),
            RecVT::Branch(ref vt) => vt.current(),
        }
    }

    fn simplify(&mut self) -> bool {
        match *self {
            RecVT::Leaf(ref mut vt) => vt.simplify(),
            RecVT::Recur(ref mut vt) => vt.simplify(),
            RecVT::Branch(ref mut vt) => vt.simplify(),
        }
    }

    fn complicate(&mut self) -> bool {
        match *self {
            RecVT::Leaf(ref mut vt) => vt.complicate(),
            RecVT::Recur(ref mut vt) => vt.complicate(),
            RecVT::Branch(ref mut vt) => vt.complicate(),
        }
    }
}

impl<L: Strategy, R: Strategy> Strategy for RecArgInner<L, R>
where R::Value: ValueTree<Value = ValueFor<L>> {
    type Value = RecVT<L, R>;

    fn new_value(&self, runner: &mut TestRunner) -> NewTree<Self> {
        match *self {
            RecArgInner::Leaf(ref s)
                => s.new_value(runner).map(RecVT::Leaf),
            RecArgInner::Recur(ref s)
                => s.new_value(runner).map(RecVT::Recur),
            RecArgInner::Branch(ref s)
                => s.new_value(runner).map(|vt| RecVT::Branch(Box::new(vt))),
        }
    }
}

opaque_strategy_wrapper! {
    /// Argument type of the `<closure>` you pass to
    /// `leaf.prop_recursive(_, _, _, <closure>)`.
    #[derive(Debug)]
    pub struct RecArg
        [<Leaf, Recur>]
        [where Leaf: Strategy, Recur: Strategy,
               Recur::Value: ValueTree<Value = ValueFor<Leaf>>]
        (RecArgInner<Leaf, Recur>) -> RecArgValueTree<Leaf, Recur>;

    /// `ValueTree` corresponding to `RecArg`.
    pub struct RecArgValueTree
        [<Leaf, Recur>]
        [where Leaf: Strategy, Recur: Strategy,
               Recur::Value: ValueTree<Value = ValueFor<Leaf>>]
        (RecVT<Leaf, Recur>) -> ValueFor<Leaf>;
}

/// Return type from `Strategy::prop_recursive()`.
pub struct Recursive<B, F> {
    pub(super) leaf: Arc<B>,
    pub(super) recurse: Arc<F>,
    pub(super) depth: u32,
    pub(super) desired_size: u32,
    pub(super) expected_branch_size: u32,
}

impl<Leaf, Recur, F> Recursive<RecArg<Leaf, Recur>, F>
where
    F: Fn(Arc<RecArg<Leaf, Recur>>) -> Recur,
    Leaf: Strategy,
    Recur: Strategy,
    Recur::Value: ValueTree<Value = ValueFor<Leaf>>,
{
    pub(super) fn new
        (leaf: Leaf, depth: u32, desired_size: u32, expected_branch_size: u32,
         recurse: F)
        -> Self
    {
        Recursive {
            leaf: Arc::new(RecArg(RecArgInner::Leaf(leaf))),
            recurse: Arc::new(recurse),
            depth, desired_size, expected_branch_size,
        }
    }
}

impl<B: fmt::Debug, F> fmt::Debug for Recursive<B, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Recursive")
            .field("leaf", &self.leaf)
            .field("recurse", &"<function>")
            .field("depth", &self.depth)
            .field("desired_size", &self.desired_size)
            .field("expected_branch_size", &self.expected_branch_size)
            .finish()
    }
}

impl<B, F> Clone for Recursive<B, F> {
    fn clone(&self) -> Self {
        Recursive {
            leaf: Arc::clone(&self.leaf),
            recurse: Arc::clone(&self.recurse),
            depth: self.depth,
            desired_size: self.desired_size,
            expected_branch_size: self.expected_branch_size,
        }
    }
}

impl<Leaf, Recur, F> Strategy for Recursive<RecArg<Leaf, Recur>, F>
where
    F: Fn(Arc<RecArg<Leaf, Recur>>) -> Recur,
    Leaf: Strategy,
    Recur: Strategy,
    Recur::Value: ValueTree<Value = ValueFor<Leaf>>,
{
    type Value = <RecArg<Leaf, Recur> as Strategy>::Value;

    fn new_value(&self, runner: &mut TestRunner) -> NewTree<Self> {
        // Since the generator is stateless, we can't implement any "absolutely
        // X many items" rule. We _can_, however, with extremely high
        // probability, obtain a value near what we want by using decaying
        // probabilities of branching as we go down the tree.
        //
        // We are given a target size S and a branch size K (branch size =
        // expected number of items immediately below each branch). We select
        // some probability P for each level.
        //
        // A single level l is thus expected to hold PlK branches. Each of
        // those will have P(l+1)K child branches of their own, so there are
        // PlP(l+1)K² second-level branches. The total branches in the tree is
        // thus (Σ PlK^l) for l from 0 to infinity. Each level is expected to
        // hold K items, so the total number of items is simply K times the
        // number of branches, or (K Σ PlK^l). So we want to find a P sequence
        // such that (lim (K Σ PlK^l) = S), or more simply,
        // (lim Σ PlK^l = S/K).
        //
        // Let Q be a second probability sequence such that Pl = Ql/K^l. This
        // changes the formulation to (lim Σ Ql = S/K). The series Σ0.5^(l+1)
        // converges on 1.0, so we can let Ql = S/K * 0.5^(l+1), and so
        // Pl = S/K^(l+1) * 0.5^(l+1) = S / (2K) ^ (l+1)
        //
        // We don't actually have infinite levels here since we _can_ easily
        // cap to a fixed max depth, so this will be a minor underestimate. We
        // also clamp all probabilities to 0.9 to ensure that we can't end up
        // with levels which are always pure branches, which further
        // underestimates size.

        let mut branch_probabilities = Vec::new();
        let mut k2 = u64::from(self.expected_branch_size) * 2;
        for _ in 0..self.depth {
            branch_probabilities.push(f64::from(self.desired_size) / k2 as f64);
            k2 = k2.saturating_mul(u64::from(self.expected_branch_size) * 2);
        }

        let mut strat = Arc::clone(&self.leaf);

        while let Some(branch_probability) = branch_probabilities.pop() {
            let recursed = (self.recurse)(Arc::clone(&strat));
            let recursive_choice = Arc::new(RecArg(RecArgInner::Recur(recursed)));

            let non_recursive_choice = strat;

            let branch_strat = ::bool::weighted(branch_probability.min(0.9));

            let branched_strat = IndFlatten(Map::new(branch_strat, BranchFn {
                yes: recursive_choice,
                no: non_recursive_choice
            }));

            strat = Arc::new(RecArg(RecArgInner::Branch(branched_strat)));
        }

        strat.new_value(runner)
    }
}

#[cfg(test)]
mod test {
    use std::cmp::max;

    use strategy::just::Just;
    use super::*;

    #[test]
    fn test_recursive() {
        #[derive(Clone, Debug)]
        enum Tree {
            Leaf,
            Branch(Vec<Tree>),
        }

        impl Tree {
            fn stats(&self) -> (u32, u32) {
                match *self {
                    Tree::Leaf => (0, 1),
                    Tree::Branch(ref children) => {
                        let mut depth = 0;
                        let mut count = 0;
                        for child in children {
                            let (d, c) = child.stats();
                            depth = max(d, depth);
                            count += c;
                        }

                        (depth + 1, count + 1)
                    }
                }
            }
        }

        let mut max_depth = 0;
        let mut max_count = 0;

        let strat = Just(Tree::Leaf).prop_recursive(4, 64, 16,
            |element| {
                ::collection::vec(element, 8..16).prop_map(Tree::Branch).boxed()
            });

        let mut runner = TestRunner::default();
        for _ in 0..65536 {
            let tree = strat.new_value(&mut runner).unwrap().current();
            let (depth, count) = tree.stats();
            assert!(depth <= 4, "Got depth {}", depth);
            assert!(count <= 128, "Got count {}", count);
            max_depth = max(depth, max_depth);
            max_count = max(count, max_count);
        }

        assert!(max_depth >= 3, "Only got max depth {}", max_depth);
        assert!(max_count > 48, "Only got max count {}", max_count);
    }
}
