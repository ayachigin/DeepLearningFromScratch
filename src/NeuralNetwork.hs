module NeuralNetwork where

import Prelude hiding (map, zipWith)
import qualified Prelude
import Data.Array.Repa hiding ((++))
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (mmultS, col)
import Data.Array.Repa.Repr.Vector (V, computeVectorP)
import System.Random (mkStdGen, randomRs, randomIO)
import Control.Lens
import Data.List.Lens

import Debug.Trace (trace)

import Mnist (NormalizedDataSets)
import Util (updateAS, modifyA, modifyL, modifyAS)

type Matrix r = Array r DIM2 Double

type Vector = Array U DIM1 Double

type Weight = Matrix U

type Bias = Matrix U

type ActivationFunction = Matrix D -> Matrix U

type Layer = (Weight, Bias, ActivationFunction)

type Gradient = ([Double], [Double])

type Gradients = [Gradient]

weight :: Layer -> Weight
weight = (^._1)

bias :: Layer -> Bias
bias = (^._2)

type NN = [Layer]

type NNs = Array V DIM1 NN

type LossFunction = Matrix U -> Matrix U -> Double

network :: [Int] -> (Double, Double) -> IO NN
network (x1:x2:xs) range = do
  n <- network'
  let newLastLayer = last n & _3 .~ softmax
  return $ n & ix (length xs) .~ newLastLayer
  where
    network' =  sequence $ scanl f y xs
    y = g x1 x2
    g :: Int -> Int -> IO Layer
    g a b = do
      r1 <- randomIO
      r2 <- randomIO
      return $ ( matrix (ix2 a b) (a*b) range r1
               , matrix (ix2 1 b) b range r2
               , sigmonoid)
    f :: IO Layer -> Int -> IO Layer
    f a b = do
      l <- a
      g (col.extent.weight $ l) b

performNN :: Monad m => NormalizedDataSets -> Int -> NN -> m Double
performNN dataset batsize net = do
  as <- computeVectorP $ map (performNNStep net) dataset
  return $ (/(fromIntegral batsize)) . sum . toList $ as
  where
    performNNStep :: NN -> (Matrix U, Int) -> Double
    performNNStep n (d, l) = loss d n crossEntropyError l
    loss :: Matrix U -> NN -> LossFunction -> Int -> Double
    loss input n lossFunc label = flip (calcLoss lossFunc) label $
                                   predict input n
    predict :: Matrix U -> NN -> Matrix U
    predict input n = foldl f input n
      where
        f :: Matrix U -> Layer -> Matrix U
        f i (w, b, a) = a .
                        zipWith (+) b $ mmultS i w

numericalGradient :: Monad m =>
                     NormalizedDataSets -> NN -> Int -> m Gradients
numericalGradient input net bat = mapM calcGradient [0..len]
  where
    len = length net - 1
    h :: Double
    h = 1e-2
    calcGradient :: Monad m => Int -> m Gradient
    calcGradient i = do
      a <- diffWeights
      b <- diffBiases
      return ( a
             , b)
      where
        thisLayer = net !! i
        shW = extent . weight $ thisLayer
        shB = extent . bias   $ thisLayer
        diffNNs :: Monad m => [(Layer, Layer)] -> m [Double]
        diffNNs = mapM f
        f :: Monad m => (Layer, Layer) -> m Double
        f (l, r) = do
          a <- performNN input bat (modifyL net i l)
          b <- performNN input bat (modifyL net i r)
          return $ (b - a) / (2*h)
        diffWeights, diffBiases :: Monad m => m [Double]
        diffWeights = let l = net !! i
                          w = weight l
                          xs = [0..(size.extent $ w) - 1]
                      in mapM (f . (diffXW l w)) xs
        diffXW :: Layer -> Weight -> Int -> (Layer, Layer)
        diffXW l w n = let newWeight1 = updateAS w n (flip (-) h)
                           newWeight2 = updateAS w n (+h)
                           newLayer1  = l & _1 .~ newWeight1
                           newLayer2  = l & _1 .~ newWeight2
                       in (newLayer1, newLayer2)
        diffBiases = let l = net !! i
                         b = bias l
                         xs = [0..(size.extent $ b) - 1]
                      in mapM (f . (diffXB l b)) xs
        diffXB :: Layer -> Bias -> Int -> (Layer, Layer)
        diffXB l b n = let newBiases1 = updateAS b n (flip (-) h)
                           newBiases2 = updateAS b n (+h)
                           newLayer1  = l & _2 .~ newBiases1
                           newLayer2  = l & _2 .~ newBiases2
                       in (newLayer1, newLayer2)

gradientDescent :: Monad m => Double -> NN -> Gradients -> m NN
gradientDescent learningRate = (mapM f .) . zip
  where
    f :: Monad m => (Layer, Gradient) -> m Layer
    f ((w, b, a), (gw, gb)) = do
      w' <- computeP $ w -^ gw'
      b' <- computeP $ b -^ gb'
      return (w', b', a)
      where
        shW = extent w
        shB = extent b
        gw', gb' :: Matrix D
        gw' = map (*learningRate) $ fromListUnboxed shW gw
        gb' = map (*learningRate) $ fromListUnboxed shB gb

{-
dummy :: Monad m => NN -> m Gradients
dummy = mapM f
  where
    f (w, b, _) = do
      w' <- computeP $ map (+0.001) w
      b' <- computeP $ map (+0.02) b
      return (w', b')
-}

{- | loass function meanSquaredError
>>> let y = fromListUnboxed (ix2 1 10) [0.1, 0.05, 0.1, 0, 0.05, 0.1, 0, 0.6, 0, 0]
>>> let t = fromListUnboxed (ix2 1 10) [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>> meanSquaredError y t
0.5974999999999999
>>> let y = fromListUnboxed (ix2 1 10) [0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0]
>>> meanSquaredError y t
9.750000000000003e-2
-}
meanSquaredError :: LossFunction
meanSquaredError x y = 0.5 * (sumAllS $ zipWith (\a b -> (a-b)^2) x y)

crossEntropyError :: LossFunction
crossEntropyError x y = (*(-1)) . sumAllS . zipWith (*) y $
                        map (log . (+delta)) x
  where
    delta = 1e-7

calcLoss :: LossFunction -> Matrix U -> Int -> Double
calcLoss f m i = f m $ arr i
  where
    arr = fromListUnboxed (ix2 1 10) . shift bits
    shift [] _ = error "Shift empty list"
    shift ls 0 = ls
    shift (x:xs) n = shift (xs ++ [x]) (n-1)
    bits = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

matrix :: DIM2 -> Int -> (Double, Double) -> Int -> Matrix U
matrix shape len range gen = fromListUnboxed shape $
                             take len . randomRs range $ mkStdGen gen

sigmonoid :: ActivationFunction
sigmonoid = computeS.map (\x -> 1 / (1 + exp (-x)))

softmax :: ActivationFunction
softmax x = let c = foldAllS max 0 x
                expA :: Matrix U
                expA = computeS $ map (\a -> exp $ a - c) x
                sumExpA = sumAllS expA
            in computeS $ map (/sumExpA) expA
