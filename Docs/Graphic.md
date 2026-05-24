# Graphic

This document describes the current pygame graphics layer for the Risk project.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run The Demo

```powershell
python demo_pygame_risk_map.py
```

To render the gameplay-style layout with a left sidebar and the full map fitted on the right:

```powershell
python demo_pygame_risk_map.py --layout gameplay
```

To save a preview without opening a window:

```powershell
python demo_pygame_risk_map.py --save Docs\pygame_port_preview.png
```

## Summary

The map is rendered in five layers:

1. Load the base map image.
2. Draw translucent colored territory polygons on top of it.
3. Blit pre-rendered territory name labels from a sprite sheet.
4. Draw continent name and bonus badges.
5. Draw army count badges as circles plus text.

The renderer lives in `Graphics/risk_map.py`, the demo-state builder lives in `Graphics/demo_state.py`, and the demo layouts plus pygame loop live in `Graphics/demo_loop.py`.

## Architecture Boundary

The graphics layer is responsible only for drawing the board and UI from game data.

- `Graphics/risk_map.py` remains the reusable board renderer.
- `Graphics/demo_loop.py` and `Graphics/demo_state.py` are graphics/demo references only. They should not grow into the real game loop or game-state model.
- The real project architecture is split into `Environment`, `Graphics`, `Agents`, and a thin `Game` loop.

The future game loop should only initialize systems, ask the current agent for an action, send the action to the environment, and render the result.

## Main Files

- `Graphics/risk_map.py`
  The actual renderer.
- `Graphics/demo_state.py`
  Builds a demo game state.
- `Graphics/demo_loop.py`
  Builds the demo layouts and runs the pygame demo loop.
- `Assets/RiskMap/risk_map_data.json`
  All map metadata in one JSON file.
- `Assets/RiskMap/map_grey_new.jpg`
  The parchment-style background map.
- `Assets/RiskMap/names.png`
  The territory-name sprite sheet.

## Using The Renderer

```python
import pygame
from Graphics import RiskMapRenderer

pygame.init()
renderer = RiskMapRenderer()

owners = {
  "Alaska": 0,
  "Ontario": 1,
  "Brazil": 3,
  "India": 4,
}

armies = {
  "Alaska": 3,
  "Ontario": 7,
  "Brazil": 12,
  "India": 5,
}

surface = renderer.render(
  owners=owners,
  armies=armies,
  target_size=(1280, 815),
)
```

`owners` should use territory ids such as `Alaska`, `Ontario`, `NorthAfrica`, and `Kamchatka`.

By default, integer owner ids `0..5` map to the six built-in player colors. You can also pass an RGB tuple instead of a player id:

```python
owners = {
  "Alaska": (80, 160, 255),
  "Brazil": (255, 140, 80),
}
```

## Data Used By The Renderer

The JSON file contains six main groups of data:

- `territory_paths`
  SVG path strings for each territory shape.
- `territory_names`
  The territory ids and display names.
- `army_points`
  The coordinates where each army badge should be drawn.
- `font_sprite_coords`
  The crop rectangle for each territory label inside `names.png`.
- `font_destination_coords`
  The on-map position where each cropped label image should be placed.
- `continents`
  The continent definitions, each with a territory list, standard reinforcement bonus value, and a map label anchor position.

## Render Flow

### 1. Renderer Setup

When `RiskMapRenderer` is created in `Graphics/risk_map.py`, it does the following:

- Resolves the asset folder path.
- Loads `risk_map_data.json`.
- Loads `map_grey_new.jpg` into `self.base_map`.
- Loads `names.png` into `self.names_sprite`.
- Precomputes cropped label surfaces.
- Precomputes polygon point lists for every territory.

This means the expensive parsing work happens once during setup, not on every frame.

### 2. Loading JSON Data

`_load_data()` reads the JSON file and returns a Python dictionary.

That is the only runtime data source for the map. There is no JS parsing anymore.

### 3. Building Territory Labels

`_build_label_surfaces()` uses:

- `font_sprite_coords` to crop a small rectangle from `names.png`
- `font_destination_coords` later to place that label on the map

So the labels are not rendered with a font. They are image snippets taken from the sprite sheet.

### 4. Building Territory Polygons

`_build_territory_polygons()` loops through `territory_paths` and sends each SVG path string into `_path_to_polygons()`.

`_path_to_polygons()` does this:

- Parses the SVG path string with `svg.path.parse_path()`.
- Splits the path into subpaths.
- Samples points along each segment.
- Converts those sampled points into pygame polygon point lists.

Straight line segments use very few samples.
Curved segments use more samples, controlled by `curve_steps`.

The result is a dictionary like this in practice:

```python
{
    "Alaska": [[(x1, y1), (x2, y2), ...]],
    "Brazil": [[...]],
}
```

Those polygons are what pygame actually fills.

## How `render()` Works

The `render()` method creates one final `pygame.Surface` and draws into it in order.

### Step 1: Start with the base map

A new transparent surface is created and the background image is blitted onto it.

```python
canvas = pygame.Surface(self.base_map.get_size(), pygame.SRCALPHA)
canvas.blit(self.base_map, (0, 0))
```

At this point, the map is only the parchment background.

### Step 2: Draw territory ownership colors

A second transparent surface called `overlay` is created.

For each territory:

- Look up the owner color from the `owners` input dictionary.
- Resolve that owner into an RGB color.
- Add the configured alpha value.
- Fill each polygon for that territory using `pygame.draw.polygon()`.

Then the overlay is blitted over the background.

The territory color is translucent, so the background texture still shows through.

### Step 3: Draw territory names

For each territory id:

- Fetch its prebuilt label image.
- Fetch its destination coordinates.
- Blit the label image onto the canvas.

The labels are drawn after the colored overlay, so they stay readable.

### Step 4: Draw continent badges

`_draw_continent_labels()` loops over the `continents` section in the JSON.

For each continent:

- Read the continent `label_position` anchor.
- Render the continent name.
- Render the bonus as `+N`.
- Draw both inside a small translucent badge.

The badge positions are data-driven, so the placement can be tuned in `Assets/RiskMap/risk_map_data.json` without changing renderer code.

In the current layout, those anchors are intentionally placed just outside each continent, usually in nearby sea, so the badges do not cover territory shapes or territory name labels.

These badges are drawn after the territory labels so they stay readable over the background texture, but before army circles so army counts remain the top-most gameplay markers.

### Step 5: Draw army badges

`_draw_armies()` loops over the `armies` dictionary.

For each territory with an army count:

- Read the badge center from `army_points`.
- Draw a light gray filled circle.
- Draw a darker border circle.
- Render the troop count text with `pygame.font.SysFont()`.
- Center the text inside the circle.

These badges are drawn last so they stay on top of both the fill layer and the labels.

### Step 6: Optional resize

If `target_size` is provided, the fully rendered surface is scaled using `pygame.transform.smoothscale()`.

That means all drawing is done at the original asset resolution first, and only then resized for display.

## Inputs To `render()`

`render()` accepts three inputs:

- `owners`
  A dictionary mapping territory ids to either a player index or an RGB tuple.
- `armies`
  A dictionary mapping territory ids to troop counts.
- `target_size`
  The output size of the final surface.

Example:

```python
surface = renderer.render(
    owners={
        "Alaska": 0,
        "Brazil": 3,
        "India": (180, 120, 255),
    },
    armies={
        "Alaska": 3,
        "Brazil": 11,
        "India": 5,
    },
    target_size=(1280, 815),
)
```

## Demo Flow

The demo code is split into two small modules.

- `Graphics/demo_state.py`
  `build_demo_state()` creates random owners and army counts.
- `Graphics/demo_loop.py`
  `run_map_demo()` creates the renderer, builds a chosen demo layout, and either:
  saves the result to disk, or
  opens a pygame window and shows the rendered surface inside a simple loop.

`Graphics/demo_loop.py` currently supports two demo layouts:

- `full`
  Renders only the map and scales the final map surface to the requested window size.
- `gameplay`
  Builds a larger UI surface with three parts:
  a top header,
  a left sidebar with mock player, card-count, and attack controls,
  and the full map fitted into the right content area.

The `gameplay` layout still renders the whole map. It does not crop into a region. Instead, it renders the map at the original asset resolution, computes the largest size that fits inside the right panel, and then smooth-scales the full map into that area.

This is handled by two helpers inside `Graphics/demo_loop.py`:

- `_render_full_map_demo()`
  Thin wrapper around `RiskMapRenderer.render()`.
- `_render_gameplay_demo()`
  Composes the header, sidebar, buttons, and fitted map preview into one final surface.

## Why This Design Is Simple

This layout is simple because it separates concerns clearly:

- `Assets/RiskMap`
  Pure data and images.
- `Graphics/risk_map.py`
  Pure rendering logic.
- `Graphics/demo_state.py`
  Demo-only fake game state.
- `Graphics/demo_loop.py`
  Demo-only layouts and pygame loop.
- `demo_pygame_risk_map.py`
  Thin command-line entry point.

## Practical Notes

- The renderer is best when reused as a cached object, because path parsing happens during initialization.
- Army circles and text are dynamic and are meant to change often.
- Territory labels are static image fragments, not font rendering.
- The final map style comes from mixing a textured background with semi-transparent polygon fills.
