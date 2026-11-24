// Landing background: Figma's "Moving gradient" shader fill (WebGPU), mounted with
// @figma/expressive-components. Parameters are the design's (Figma node 15:6).
// Browsers without WebGPU render nothing here, leaving the plain black background.
import React from 'react';
import { createRoot } from 'react-dom/client';
import { ShaderFill } from '@figma/expressive-components';
import {
  setup as setupCodeComponentId_8ce92017e53431a2f04b3574f4ba7c98f6f55f1e_8358,
  render as renderCodeComponentId_8ce92017e53431a2f04b3574f4ba7c98f6f55f1e_8358,
  manifest as manifestCodeComponentId_8ce92017e53431a2f04b3574f4ba7c98f6f55f1e_8358,
} from './lib/custom-effects/CodeComponentId_8ce92017e53431a2f04b3574f4ba7c98f6f55f1e_8358.js';

const shader = {
  setup: setupCodeComponentId_8ce92017e53431a2f04b3574f4ba7c98f6f55f1e_8358,
  render: renderCodeComponentId_8ce92017e53431a2f04b3574f4ba7c98f6f55f1e_8358,
  params: {
    intensity: 3.569999933242798,
    gradient: {
      stops: [
        // Moth palette: one signal accent (#DCF900) with support gray (#E7E7E7) on black
        { position: 0, color: { r: 0, g: 0, b: 0, a: 1 } },
        { position: 0.20673076808452606, color: { r: 220 / 255, g: 249 / 255, b: 0, a: 1 } },
        { position: 0.4423076808452606, color: { r: 0, g: 0, b: 0, a: 1 } },
        { position: 0.567307710647583, color: { r: 231 / 255, g: 231 / 255, b: 231 / 255, a: 1 } },
        { position: 0.7403846383094788, color: { r: 0, g: 0, b: 0, a: 1 } },
        { position: 0.9615384340286255, color: { r: 220 / 255, g: 249 / 255, b: 0, a: 1 } },
      ],
    },
    gradientBalance: 0,
    material: 0,
    morphSpeed: 0.6100000143051147,
    detail: 1.96999990940094,
    twist: 1.0099999904632568,
    zoom: 80,
    gradientMethod: 0,
    warp: 0.25999999046325684,
    rotationSpeed: 6,
  },
  manifest: manifestCodeComponentId_8ce92017e53431a2f04b3574f4ba7c98f6f55f1e_8358,
};

/**
 * Render the landing shader fill into an element (it fills the element), only while the
 * landing screen shows. ShaderFill has no pause: its WebGPU loop draws every frame for as
 * long as it is mounted, so off the landing (where the element is at opacity 0) it is
 * unmounted, and mounted again on the way back.
 *
 * Args:
 *   el: Positioned container; the design shows it at 26 % opacity.
 *   screenOf: The element whose ``data-screen`` attribute names the screen ("landing", ...).
 */
export function mountShaderBackground(el, screenOf) {
  let root = null;
  const sync = () => {
    const want = screenOf.dataset.screen === 'landing';
    if (want && !root) {
      root = createRoot(el);
      root.render(React.createElement(ShaderFill, {
        shader, style: { position: 'absolute', inset: 0, width: '100%', height: '100%' },
      }));
    } else if (!want && root) {
      root.unmount();
      root = null;
    }
  };
  new MutationObserver(sync).observe(screenOf, { attributes: true, attributeFilter: ['data-screen'] });
  sync();
}
