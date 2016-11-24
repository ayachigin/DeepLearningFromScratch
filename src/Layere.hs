{-# LANGUAGE FlexibleContexts #-}
module Main where

import Prelude hiding (map, traverse)
import Control.Monad.Trans.State
import Data.Array.Repa
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

class Layer a where
  forward  :: a -> (Array D DIM2 Double, a)
  backward :: a -> Array U DIM2 Double ->
              (Array U DIM2 Double, a)

data Affine = Affine { input :: Array U DIM2 Double
                     , weight :: Array U DIM2 Double
                     , bias   :: Array U DIM2 Double
                     } deriving (Show, Read)

instance Layer Affine where
  forward a@(Affine x w b) = ((mmultS x w) +^^ b, a)
  backward (Affine x w _) dout = let dx = mmultS dout (ts w)
                                     dw = mmultS (ts x) dout
                                     db = mapSum dout
                                 in (dx, Affine dx dw db)
    where ts = computeS . transpose

newtype Sigmonoid = Sigmonoid (Array D DIM2 Double)

instance Layer Sigmonoid where
  forward (Sigmonoid input) = (v, Sigmonoid v)
      where
        v = map f input
        f x = 1 / (1 + exp(-x))
  backward (Sigmonoid out) dout = let dx = dout *^ (map (1.0-) out) *^ out
                                  in (computeS dx, Sigmonoid dx)

newtype SoftMaxWithLoss = SoftMaxWithLoss ( Double
                                          , Array U DIM2 Double
                                          , Array U DIM2 Double)

instance Layer SoftMaxWithLoss where
  forward (SoftMaxWithLoss (_, x, t)) = let y = softmax x
                                            loss = crossEntropyError y t
                                        in (delay x ,SoftMaxWithLoss (loss, y, t))

softmax :: Array U DIM2 Double -> Array U DIM2 Double
softmax = undefined

crossEntropyError :: Matrix -> Matrix -> Double
crossEntropyError = undefined

(+^^) :: (Source r1 Double, Source r2 Double) =>
      Array r1 DIM2 Double -> Array r2 DIM2 Double -> Array D DIM2 Double
(+^^) x b = if r == 1 then
              fromFunction shx f
            else
              x +^ b
  where
    r = row . extent $ b
    shx = extent x
    f :: DIM2 -> Double
    f sh = (index x sh) + (g sh)
    g :: DIM2 -> Double
    g sh = index b (ix2 0 c)
      where c = col sh

foldRow :: (Double -> Double -> Double) -> Double -> Array U DIM2 Double ->
           Array D DIM1 Double
foldRow f acc arr = traverse arr (\(Z:.x:._) -> ix1 x)
                    (\find (Z:.i) -> foldl f acc [g i j | j <- [0..y-1]])
  where
    g x y = index arr (ix2 x y)
    (Z:.x:.y) = extent arr

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
reLUF = fmap (fmap (\x -> if x > 0 then x else 0)) get

reLUB :: BackwardL
reLUB douts = do
  inputs <- get
  put $ fmap (\(x, y) -> if x > 0 then y else 0) . zip inputs $ douts

sigmonoidF :: ForwardL
sigmonoidF = do
  ls <- get
  let out = fmap f ls
  put out
  return out
  where f x = 1 / (1 + exp(-x))

sigmonoidB :: BackwardL
sigmonoidB douts = do
  outs <- get
  put $ fmap f . zip douts $ outs
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

softMaxWithLossF :: (Matrix, Matrix) ->
                    ( Double
                    , (Matrix, Matrix))
softMaxWithLossF (x, t) = let loss = calcLoss x t
                          in (loss, (x, t))

softMaxWithLossB :: (Matrix, Matrix) ->
                    Array D DIM2 Double
softMaxWithLossB (x, t) = map (/(fromIntegral s)) $ x -^ t
  where
    s = size . extent $ x

calcLoss :: Matrix -> Matrix -> Double
calcLoss = undefined

mapSum :: Matrix -> Matrix
mapSum xs = fromListUnboxed (ix2 1 c) $ fmap f [0..c-1]
  where sh = extent xs
        c = col sh
        r = row sh
        f x = foldl (g x) 0 [0..r-1]
        g x acc y = acc + (xs `index` ix2 x y)
