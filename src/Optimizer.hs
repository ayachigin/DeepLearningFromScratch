module Optimizer where

import Prelude hiding (map)
import Data.Array.Repa hiding (zipWith)

import Types

type Momentum = [(Weight, Bias)]

type AdamGrad = [(Weight, Bias)]

moment :: Double
moment = 0.9

updateMomentumMaybe :: Double -> Double -> Maybe Momentum -> NN -> Gradients ->
                       (NN, Momentum)
updateMomentumMaybe m lr vs n gs = case vs of
                                     (Just vs') ->
                                       updateMomentum m lr vs' n gs
                                     Nothing ->
                                       updateMomentum m lr evs n gs
  where
    evs = fmap f n
    f (w, b, _) = ( computeS $ map (const 0) w
                  , computeS $ map (const 0) b)

updateMomentum :: Double -> Double -> Momentum -> NN -> Gradients ->
                  (NN, Momentum)
updateMomentum m lr vs n gs = (newNN, m')
  where
    newNN = zipWith f n m'
    f (w, b, fn) (mw, mb) = ( computeS $ w +^ mw
                           , computeS $ b +^ mb
                           , fn)
    m' = zipWith g vs gs
    g :: (Weight, Bias) -> ([Double], [Double]) -> (Weight, Bias)
    g (vw, vb) (gw, gb) = ( computeS $ (map (*m) vw) -^ (map (*lr) gwa)
                          , computeS $ (map (*m) vb) -^ (map (*lr) gba))
      where
        gwa = fromListUnboxed (extent vw) gw
        gba = fromListUnboxed (extent vb) gb

updateAdamGradMaybe :: Double -> Maybe AdamGrad -> NN -> Gradients ->
                       (NN, AdamGrad)
updateAdamGradMaybe lr (Just vs) n gs = updateAdamGrad lr vs n gs
updateAdamGradMaybe lr Nothing   n gs = updateAdamGrad lr evs n gs
  where evs = fmap f n
        f (w, b, _) = ( computeS $ map (const 0) w
                      , computeS $ map (const 0) b)

updateAdamGrad :: Double -> AdamGrad -> NN -> Gradients ->
                  (NN, AdamGrad)
updateAdamGrad lr vs n gs = (newNN, h)
  where
    newNN = zipWith f n $ zip gs h
    f :: Layer -> (Gradient, (Weight, Bias)) -> Layer
    f (w, b, fn) ((gw, gb), (mw, mb)) = ( computeS $ k w gwa mw
                                        , computeS $ k b gba mb
                                        , fn)
      where
        gwa = fromListUnboxed (extent mw) gw
        gba = fromListUnboxed (extent mb) gb
        k :: Matrix U -> Matrix U -> Matrix U -> Matrix D
        k w g m = w -^ map (*lr) (g /^ map ((+1e-4).sqrt) m)
    h = zipWith g vs $ fmap square gs
    square (w, b) = (fmap (^2) w, fmap (^2) b)
    g :: (Weight, Bias) -> ([Double], [Double]) -> (Weight, Bias)
    g (vw, vb) (gw, gb) = ( computeS $ vw +^ gwa
                          , computeS $ vb +^ gba)
      where
        gwa = fromListUnboxed (extent vw) gw
        gba = fromListUnboxed (extent vb) gb
