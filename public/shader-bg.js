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
        { position: 0, color: { r: 0, g: 0, b: 0, a: 1 } },
        { position: 0.20673076808452606, color: { r: 0.47058823704719543, g: 1, b: 0.9911764860153198, a: 1 } },
        { position: 0.4423076808452606, color: { r: 0, g: 0, b: 0, a: 1 } },
        { position: 0.567307710647583, color: { r: 0.46514421701431274, g: 0.7504005432128906, b: 1, a: 1 } },
        { position: 0.7403846383094788, color: { r: 0, g: 0, b: 0, a: 1 } },
        { position: 0.9615384340286255, color: { r: 0, g: 1, b: 0.6500000357627869, a: 1 } },
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
 * Render the landing shader fill into an element (it fills the element).
 * @param {HTMLElement} el - Positioned container; the design shows it at 26 % opacity.
 */
export function mountShaderBackground(el) {
  createRoot(el).render(React.createElement(ShaderFill, {
    shader, style: { position: 'absolute', inset: 0, width: '100%', height: '100%' },
  }));
}
