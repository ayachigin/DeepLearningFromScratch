{-# LANGUAGE TypeOperators #-}
module CNN where

import Data.Array.Repa

import Types

data Forward

data Backward

type MiniBatchInput = Array D DIM4 Double

data Convolution = Convolution (Matrix D) (Matrix D) Int Int

data CNNLayer a = ConvolutionLayer Convolution
                | Pooling

data Im2ColOption = Im2ColOption
  { inputSize    :: Int
  , filterHeight :: Int
  , filterWidth  :: Int
  , strideSize   :: Int
  , padSize      :: Int
  } deriving (Show, Read, Eq)

im2col :: (Source r1 e, Source r2 e, Shape sh) =>
          Im2ColOption ->
          Array r1 sh e ->
          Array r2 DIM2 e
im2col = undefined

convolutionForward :: MiniBatchInput ->
                      Convolution ->
                      CNNLayer Backward
convolutionForward = undefined

-- | focus
--
-- It provides access to the array that reduced one dimension of
-- muitidimensional array by index.
focus :: (Source r e, Shape sh) =>
         (Array r (sh :. Int) e) ->
         Int ->
         Array D sh e
focus arr idx = fromFunction sh' f
  where (sh' :. _) = extent arr
        f sh = index arr (sh :. idx)
