module Main where

import Control.Monad.Trans.State
import Data.Array.Repa hiding (map)
import Data.Array.Repa.Algorithms.Matrix


type Forward s a = State s a

type Backward s a = a -> State s ()

type ForwardB = Forward (Double, Double) Double

type BackwardB = Backward (Double, Double) Double

type ForwardL = Forward [Double] [Double]

type BackwardL = Backward [Double] [Double]

type ForwardA r sh = Forward (Array r sh Double)
                             (Array r sh Double)

type Matrix = Array U DIM2 Double

main :: IO ()
main = do
  let arr = fromListUnboxed (ix2 2 3) [(1::Int)..6]
      c :: Array U DIM2 Int
      c = computeS $ transpose arr
  print c

runForward :: (State (a, b) c) -> a -> b -> (c, (a, b))
runForward f x y = runState f (x, y)

runBackward :: State s a -> (t, s) -> (a, s)
runBackward f (_, s) = runState f s

multF :: ForwardB
multF = do
  (x, y) <- get
  return (x * y)

multB :: BackwardB
multB dout = do
  (x, y) <- get
  put (dout*y, dout * x)

addF :: ForwardB
addF = fmap (uncurry (+)) get

addB :: BackwardB
addB d = put (d, d)

reLUF :: ForwardL
reLUF = fmap (map (\x -> if x > 0 then x else 0)) get

reLUB :: BackwardL
reLUB douts = do
  inputs <- get
  put $ map (\(x, y) -> if x > 0 then y else 0) . zip inputs $ douts

sigmonoidF :: ForwardL
sigmonoidF = do
  ls <- get
  let out = map f ls
  put out
  return out
  where f x = 1 / (1 + exp(-x))

sigmonoidB :: BackwardL
sigmonoidB douts = do
  outs <- get
  put $ map f . zip douts $ outs
  where f (d, o) = d * (1.0 - o) * o
