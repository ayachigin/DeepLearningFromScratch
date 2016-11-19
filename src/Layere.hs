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

affineF :: (Matrix, Matrix, Matrix) ->
           ( Array D DIM2 Double,
            (Matrix, Matrix, Matrix))
affineF (w, b, x) = ((mmultS x w) +^ b, (w, b, x))

affineFS :: Monad m =>
            StateT (Matrix, Matrix, Matrix) m (Array D DIM2 Double)
affineFS = state affineF

affineB :: Source r Double =>
           Matrix ->
           ( Array r DIM2 Double, t, Array r DIM2 Double) ->
           ((), (Matrix, Matrix, Matrix))
affineB dout (w, _, x) = let dX = mmultS dout (ts w)
                             dW = mmultS (ts x) dout
                             dB = mapSum dout
                         in ((), (dB, dW, dX))
  where ts = computeS . transpose

affineBS :: Monad m =>
            Matrix ->
            StateT (Matrix, Matrix, Matrix) m ()
affineBS = \x -> state (affineB x)

mapSum :: Matrix -> Matrix
mapSum xs = fromListUnboxed (ix2 1 c) $ map f [0..c-1]
  where sh = extent xs
        c = col sh
        r = row sh
        f x = foldl (g x) 0 [0..r-1]
        g x acc y = acc + (xs `index` ix2 x y)
