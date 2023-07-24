'use strict';

// LOW LEVEL UTILS

const create_element = function(tag, options, ...children) {
  const el = document.createElement(tag);
  options = options || {};
  for (const k in options)
    if (k === 'events')
      for (name in options[k])
        el.addEventListener(name, options[k][name]);
    else if (k === 'dataset')
      for (name in options[k])
        el.dataset[name] = options[k][name];
    else
      el.setAttribute(k, options[k]);
  for (const child of children)
    el.appendChild(typeof(child) === 'string' ? document.createTextNode(child) : child);
  return el;
};

const create_svg_element = function(tag, options, ...children) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  options = options || {};
  for (const k in options)
    if (k === 'events')
      for (name in options[k])
        el.addEventListener(name, options[k][name]);
    else
      el.setAttribute(k, options[k]);
  for (const child of children)
    el.appendChild(child);
  return el;
};

const union_dict = function(...elements) {
  if (elements.length == 1)
    return elements[0];
  const union = {};
  for (const elem of elements)
    Object.assign(union, elem);
  return union;
};

// ART

const create_lock = function(options) {
  const hw = 6;  // half width
  const hh = 5;  // half height
  const so = 5;  // inner radius of the shackle
  const si = 4;  // outer radius of the shackle
  const sh = 2;  // length of the straight part of the shackle
  const su = 5;  // length of the straight part of the shackle unlocked
  return (
    create_svg_element('svg', union_dict({viewBox: '-16 -18 32 32'}, options),
      create_svg_element('path', {'class': 'show-if-theater-locked', stroke: 'none', d: `M ${-hw},${-hh} v ${2*hh} h ${2*hw} v ${-2*hh} h ${so-hw} v ${-sh} a ${so} ${so} 0 0 0 ${-2*so} 0 v ${sh} Z M ${-si} ${-hh} v ${-sh} a ${si} ${si} 0 0 1 ${2*si} 0 v ${sh} Z`}),
      create_svg_element('path', {'class': 'show-if-theater-unlocked', stroke: 'none', d: `M ${-hw},${-hh} v ${2*hh} h ${2*hw} v ${-2*hh} h ${so-hw} v ${-su} a ${si} ${si} 0 0 1 ${2*si} 0 v ${su} h ${so-si} v ${-su} a ${so} ${so} 0 0 0 ${-2*so} 0 v ${su} Z`})));
};

const log_level_paths = {
  0: 'M -4 -4 L 2 -4 L 4 -2 L 4 2 L 2 4 L -4 4 Z',
  1: 'M -2 -4 L 2 -4 M 0 -4 L 0 4 M -2 4 L 2 4',
  2: 'M -4 -4 L -4 4 L 4 4 L 4 -4',
  3: 'M -4 -4 L -4 4 L 4 4 L 4 -4 M 0 4 L 0 -1',
  4: 'M 4 -4 L -4 -4 L -4 4 L 4 4 M -4 0 L 1 0'
};

const create_log_level_icon = function(level, options) {
  return (
    create_svg_element('svg', union_dict({viewBox: '-9 -9 18 18', style: 'width: 18px; height: 18px; border-radius: 1px;'}, options),
      create_svg_element('mask', {id: 'm'},
        create_svg_element('path', {d: 'M -10 -10 L -10 10 L 10 10 L 10 -10 Z', fill: 'white', stroke: 'none'}),
        create_svg_element('path', {d: log_level_paths[level], fill: 'none', 'stroke-width': '2', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', stroke: 'black'})),
      create_svg_element('path', {mask: 'url(#m)', d: 'M -9 -9 L -9 9 L 9 9 L 9 -9 Z', stroke: 'none'})));
};

// LOG

// NOTE: This should match the log levels defined in the `treelog` module.
const LEVELS = ['debug', 'info', 'user', 'warning', 'error'];
const VIEWABLE = /[.](jpg|jpeg|png|svg)$/;

const Log = class {
  constructor() {
    this.root = document.getElementById('log');
  }
  get state() {
    return {collapsed: this.collapsed, loglevel: this.loglevel};
  }
  set state(state) {
    // We deliberately ignore state changes, except during reloads (handled by
    // `init_elements` and `set loglevel` in the `window`s load event handler,
    // respectively).
  }
  get collapsed() {
    const collapsed = {};
    for (const context of document.querySelectorAll('#log .context.collapsed'))
      collapsed[context.dataset.id] = true;
    return collapsed;
  }
  get loglevel() {
    return parseInt(document.body.dataset.loglevel || 2);
  }
  set loglevel(level) {
    level = Math.max(0, Math.min(LEVELS.length-1, level));
    for (let i = 0; i < LEVELS.length; i++)
      document.body.classList.toggle('hide'+i, i < level);
    document.body.dataset.loglevel = level;
    const indicator = document.getElementById('log-level-indicator');
    if (indicator) {
      indicator.innerHTML = '';
      indicator.appendChild(create_log_level_icon(level));
    }
    if (this.state.loglevel !== level) {
      this.state.loglevel = level;
      update_state();
    }
  }
  keydown(ev) {
    if (ev.altKey || ev.ctrlKey || ev.metaKey)
      return false;
    else if (ev.key.toLowerCase() == 'c') { // Collapse all.
      for (const context of document.querySelectorAll('#log .context'))
        if (context.lastElementChild && context.lastElementChild.classList.contains('end'))
          context.classList.add('collapsed');
      update_state();
    }
    else if (ev.key.toLowerCase() == 'e') { // Expand all.
      for (const context of document.querySelectorAll('#log .context'))
        context.classList.remove('collapsed');
      update_state();
    }
    else if (ev.key == '+' || ev.key == '=') { // Increase verbosity = decrease loglevel.
      this.loglevel = this.loglevel-1;
      update_state();
    }
    else if (ev.key == '-') { // Decrease verbosity = increase loglevel.
      this.loglevel = this.loglevel+1;
      update_state();
    }
    else
      return false;
    return true;
  }
  init_elements(collapsed) {
    // Assign unique ids to context elements, collapse contexts according to
    // `state`.
    {
      let icontext = 0;
      for (const context of document.querySelectorAll('#log .context')) {
        context.dataset.label = (context.parentElement.parentElement.dataset.label || '') + context.firstChild.innerText + '/';
        context.dataset.id = icontext;
        context.classList.toggle('collapsed', collapsed[icontext] || false);
        icontext += 1;
      }
    }

    // Assign (highest) log levels of children to context: loop over all items
    // and assign the item log level to parent context elements until the context
    // not has a higher level.
    for (const item of document.querySelectorAll('#log .item')) {
      const loglevel = parseInt(item.dataset.loglevel);
      let parent = item.parentElement;
      if (parent)
        parent = parent.parentElement;
      // NOTE: `parseInt` returns `NaN` if the `parent` loglevel is undefined and
      // `NaN < loglevel` is false.
      while (parent && parent.classList.contains('context') && !(parseInt(parent.dataset.loglevel) >= loglevel)) {
        parent.dataset.loglevel = loglevel;
        parent = parent.parentElement;
        if (parent)
          parent = parent.parentElement;
      }
    }

    // Link viewable anchors to theater.
    let ianchor = 0;
    for (const anchor of document.querySelectorAll('#log .item > a')) if (VIEWABLE.test(anchor.download)) {
      anchor.classList.add('viewable');
      anchor.addEventListener('click', this._plot_clicked);
      anchor.id = `plot-${ianchor}`;
      ianchor += 1;
      theater.add_plot(anchor);
    }

    // Make contexts clickable.
    for (const title of document.querySelectorAll('#log .context > .title'))
      title.addEventListener('click', this._context_toggle_collapsed);
  }
  _plot_clicked(ev) {
    ev.stopPropagation();
    ev.preventDefault();
    window.history.pushState(window.history.state, 'log');
    theater.anchor = ev.currentTarget;
    document.body.dataset.show = 'theater';
    update_state();
  }
  _context_toggle_collapsed(ev) {
    // `ev.currentTarget` is the context title element (see https://developer.mozilla.org/en-US/docs/Web/API/Event/currentTarget)
    const context = ev.currentTarget.parentElement;
    context.classList.toggle('collapsed');
    update_state();
    ev.stopPropagation();
    ev.preventDefault();
  }
  scroll_into_view(anchor_id) {
    const anchor = document.getElementById(anchor_id);
    if (anchor) {
      let parent = anchor.parentElement;
      while (parent && parent.id != 'log') {
        if (parent.classList.contains('context'))
          parent.classList.remove('collapsed');
        parent = parent.parentElement;
      }
      anchor.scrollIntoView();
      update_state();
    }
  }
};

// THEATER

const Theater = class {
  constructor() {
    this.root = create_element('div', {id: 'theater', events: {'pointerdown': this.pointerdown.bind(this), 'pointerup': this.pointerup.bind(this)}});
    this.plots_per_category = {undefined: []};
    this.touch_scroll_delta = 25;
  }
  add_plot(anchor) {
    anchor.dataset.label = (anchor.parentElement.parentElement.parentElement.dataset.label || '') + anchor.download;
    anchor.dataset.index = this.plots_per_category[undefined].length;
    this.plots_per_category[undefined].push(anchor);
    if (!this.plots_per_category[anchor.download])
      this.plots_per_category[anchor.download] = [];
    anchor.dataset.index_category = this.plots_per_category[anchor.download].length;
    this.plots_per_category[anchor.download].push(anchor);
  }
  get context_plots() {
    return this.anchor.parentElement.parentElement.querySelectorAll(':scope > .item > a.viewable');
  }
  get locked() {
    return document.body.classList.contains('theater-locked');
  }
  set locked(locked) {
    if (locked === undefined)
      return;
    locked = Boolean(locked);
    if (this.locked == locked)
      return;
    document.body.classList.toggle('theater-locked', locked);
    update_state();
  }
  toggle_locked() {
    document.body.classList.toggle('theater-locked');
  }
  get overview() {
    return this.root.classList.contains('overview');
  }
  set overview(overview) {
    if (overview === undefined)
      return;
    overview = Boolean(overview);
    if (this.root.classList.contains('overview') == overview)
      return;
    if (overview)
      this._draw_overview();
    else
      this._draw_plot();
    this._update_selection();
    update_state();
  }
  toggle_overview() { this.overview = !this.overview; }
  get category() {
    return this.anchor ? this.anchor.download : undefined;
  }
  get index() {
    return this.anchor && parseInt(this.locked ? this.anchor.dataset.index_category : this.anchor.dataset.index);
  }
  get anchor() {
    return this._anchor;
  }
  set anchor(anchor) {
    if (anchor === undefined || this._anchor == anchor)
      return;
    const old_anchor = this._anchor;
    this._anchor = anchor;
    if (! this.overview)
      this._draw_plot();
    else if (anchor.parentElement.parentElement != old_anchor.parentElement.parentElement)
      this._draw_overview();
    this._update_selection();
    document.getElementById('theater-label').innerText = this._anchor.dataset.label;
    update_state();
  }
  _draw_plot() {
    const plot = create_element('img', {src: this.anchor.href, 'class': 'plot', dataset: {plot_id: this.anchor.id}, events: {click: this._blur_plot.bind(this)}});
    this.root.innerHTML = '';
    this.root.classList.remove('overview');
    this.root.appendChild(plot);
  }
  _draw_overview() {
    this.root.innerHTML = '';
    this.root.classList.add('overview');
    this._update_overview_layout();
    for (const anchor of this.context_plots) {
      const plot = create_element('img', {src: anchor.href, 'class': 'plot', dataset: {plot_id: anchor.id}, events: {click: this._focus_plot.bind(this)}});
      const plot_container3 = create_element('div', {'class': 'plot_container3'}, plot);
      const plot_container2 = create_element('div', {'class': 'plot_container2'}, plot_container3);
      plot_container2.appendChild(create_element('div', {'class': 'label'}, anchor.download));
      this.root.appendChild(create_element('div', {'class': 'plot_container1'}, plot_container2));
    }
  }
  _update_selection() {
    const category = this.category;
    for (const plot of this.root.querySelectorAll('img.plot')) {
      plot.classList.toggle('selected', plot.dataset.plot_id == this.anchor.id);
    }
  }
  _update_overview_layout() {
    let nplots;
    try {
      nplots = this.context_plots.length;
    } catch (e) {
      return;
    }
    const plot_aspect = 640 / 480;
    const screen_width = window.innerWidth;
    const screen_height = window.innerHeight;
    let optimal_nrows = 1;
    let optimal_size = 0;
    for (let nrows = 1; nrows <= nplots; nrows += 1) {
      const ncols = Math.ceil(nplots / nrows);
      const size = Math.min(screen_width*screen_width/(ncols*ncols)/plot_aspect, screen_height*screen_height/(nrows*nrows)*plot_aspect);
      if (size > optimal_size) {
        optimal_nrows = nrows;
        optimal_size = size;
      }
    }
    let optimal_ncols = Math.ceil(nplots / optimal_nrows);
    this.root.style.gridTemplateColumns = Array(optimal_ncols).fill('1fr').join(' ');
    this.root.style.gridTemplateRows = Array(optimal_nrows).fill('1fr').join(' ');
  }
  _focus_plot(ev) {
    this.anchor = document.getElementById(ev.currentTarget.dataset.plot_id);
    this.overview = false;
    ev.preventDefault();
    ev.stopPropagation();
  }
  _blur_plot(ev) {
    this.overview = true;
    ev.preventDefault();
    ev.stopPropagation();
  }
  get current_plots() {
    return this.plots_per_category[this.locked && this.category || undefined];
  }
  next() {
    this.anchor = this.current_plots[this.index+1];
  }
  previous() {
    this.anchor = this.current_plots[this.index-1];
  }
  first() {
    this.anchor = this.current_plots[0];
  }
  last() {
    const plots = this.current_plots;
    this.anchor = plots[plots.length-1];
  }
  get state() {
    return {id: this.anchor.id, locked: this.locked, overview: this.overview};
  }
  set state(state) {
    if (state === undefined)
      return;
    this.anchor = document.getElementById(state.id);
    if (state.locked !== undefined)
      this.locked = state.locked;
    if (state.overview !== undefined)
      this.overview = state.overview;
  }
  _open_log() {
    document.body.dataset.show = '';
    update_state(true);
    log.scroll_into_view(this.anchor.dataset.anchor_id);
  }
  keydown(ev) {
    if (ev.altKey || ev.ctrlKey || ev.metaKey)
      return false;
    else if (ev.key == ' ')
      this.locked = !this.locked;
    else if (ev.key == 'Tab')
      this.overview = !this.overview;
    else if (ev.key == 'ArrowLeft' || ev.key == 'PageUp' || ev.key.toLowerCase() == 'k')
      this.previous();
    else if (ev.key == 'ArrowRight' || ev.key == 'PageDown' || ev.key.toLowerCase() == 'j')
      this.next();
    else if (ev.key == 'Home' || ev.key == '^')
      this.first();
    else if (ev.key == 'End' || ev.key == '$')
      this.last();
    else if (ev.key == 'Escape')
      window.history.back();
    else if (ev.key.toLowerCase() == 'q')
      this._open_log();
    else
      return false;
    return true;
  }
  pointerdown(ev) {
    if (ev.pointerType != 'touch' || !ev.isPrimary)
      return;
    this._touch_scroll_pos = ev.screenY;
    // NOTE: This introduces a cyclic reference.
    this._pointer_move_handler = this.pointermove.bind(this);
    this.root.addEventListener('pointermove', this._pointer_move_handler);
  }
  pointermove(ev) {
    if (ev.pointerType != 'touch' || !ev.isPrimary)
      return;
    if (Math.abs(ev.screenY-this._touch_scroll_pos) > this.touch_scroll_delta) {
      if (ev.screenY < this._touch_scroll_pos - this.touch_scroll_delta) {
        const delta_index = Math.floor((this._touch_scroll_pos-ev.screenY) / this.touch_scroll_delta);
        const index = Math.max(0, this.index - delta_index);
        this._touch_scroll_pos = index == 0 ? ev.screenY : this._touch_scroll_pos - delta_index*this.touch_scroll_delta;
        this.anchor = this.current_plots[index];
      }
      else if (ev.screenY > this._touch_scroll_pos + this.touch_scroll_delta) {
        const delta_index = Math.floor((ev.screenY-this._touch_scroll_pos) / this.touch_scroll_delta);
        const max_index = this.current_plots.length - 1;
        const index = Math.min(max_index, this.index + delta_index);
        this._touch_scroll_pos = index == max_index ? ev.screenY : this._touch_scroll_pos + delta_index*this.touch_scroll_delta;
        this.anchor = this.current_plots[index];
      }
    }
  }
  pointerup(ev) {
    if (ev.pointerType != 'touch' || !ev.isPrimary)
      return;
    this._touch_scroll_pos = undefined;
    this.root.removeEventListener('pointermove', this._pointer_move_handler);
  }
};

// GLOBAL

// Disabled during initialization.  Will be enabled by the window load event
// handler.
let state_control = 'disabled';

const update_state = function(push) {
  if (state_control == 'disabled')
    return;
  let state;
  if (document.body.dataset.show == 'theater')
    state = {show: 'theater', theater: theater.state};
  else
    state = {show: '', log: log.state}
  if (push)
    window.history.pushState(state, 'log');
  else
    window.history.replaceState(state, 'log');
}

const apply_state = function(state) {
  const _state_control = state_control;
  state_control = 'disabled';
  if (state.show == 'theater')
    theater.state = state.theater;
  if (state.log)
    log.state = state.log;
  document.body.dataset.show = state.show || '';
  state_control = _state_control;
  // The collapsed state is not changed by going back or forward in the
  // history.  We do store the collapsed state in `window.history.state` to
  // preserve the collapsed state during a reload.  We call `update_state` here
  // because the restored state might have a different collapsed state.
  update_state();
}

const keydown_handler = function(ev) {
  if (ev.key == 'Escape' && document.body.classList.contains('droppeddown'))
    document.body.classList.remove('droppeddown');
  else if (document.body.dataset.show == 'theater' && theater.keydown(ev))
    ;
  else if (!document.body.dataset.show && log.keydown(ev))
    ;
  else if (ev.altKey || ev.ctrlKey || ev.metaKey)
    return;
  else if (ev.key == '?')
    document.body.classList.toggle('droppeddown');
  else if (ev.key.toLowerCase() == 'r') { // Reload.
    window.location.reload(true);
  }
  else if (ev.key.toLowerCase() == 'l') { // Load latest.
    if (document.body.dataset.latest)
      window.location.href = document.body.dataset.latest + '?' + Date.now();
  }
  else
    return;
  ev.stopPropagation();
  ev.preventDefault();
}

window.addEventListener('load', function() {
  const grid = create_element('div', {'class': 'key_description'});
  const _add_key_description = function(cls, keys, description, _key) {
    grid.appendChild(create_element('div', {'class': cls+' keys', events: {click: ev => { ev.stopPropagation(); ev.preventDefault(); window.dispatchEvent(new KeyboardEvent('keydown', {key: _key})); }}}, keys.join('+')));
    grid.appendChild(create_element('div', {'class': cls}, description));
  }
  _add_key_description('', ['R'], 'Reload.', 'R');
  _add_key_description('', ['L'], 'Load latest.', 'L');
  _add_key_description('show-if-log', ['+'], 'Increase log verbosity.','+');
  _add_key_description('show-if-log', ['-'], 'Decrease log verbosity.','-');
  _add_key_description('show-if-log', ['C'], 'Collapse all contexts.','C');
  _add_key_description('show-if-log', ['E'], 'Expand all contexts.', 'E');
  _add_key_description('show-if-theater', ['TAB'], 'Toggle between overview and focus.', 'Tab');
  _add_key_description('show-if-theater', ['SPACE'], 'Lock to a plot category or unlock.', ' ');
  _add_key_description('show-if-theater', ['LEFT'], 'Show the next plot.', 'ArrowLeft');
  _add_key_description('show-if-theater', ['RIGHT'], 'Show the previous plot.', 'ArrowRight');
  _add_key_description('show-if-theater', ['Q'], 'Open the log at the current plot.', 'Q');
  _add_key_description('show-if-theater', ['ESC'], 'Go back.', 'Escape');

  var bar = document.getElementById('bar');
  // labels, only one is visible at a time
  document.getElementById('text').appendChild(create_element('div', {id: 'theater-label', 'class': 'show-if-theater hide-if-droppeddown button label', title: 'exit theater and open log here', events: {click: ev => { ev.stopPropagation(); ev.preventDefault(); theater._open_log();}}}));
  document.getElementById('text').appendChild(create_element('div', {'class': 'show-if-droppeddown label'}, 'keyboard shortcuts'));
  // log level indicator, visible in log mode
  bar.appendChild(create_element('div', {'class': 'show-if-log icon small-icon-container', id: 'log-level-indicator'}));
  // category lock button, visible in theater mode
  bar.appendChild(create_lock({'class': 'show-if-theater button icon lock', events: {click: ev => { ev.stopPropagation(); ev.preventDefault(); theater.toggle_locked(); }}}));
  // hamburger
  bar.appendChild(create_element('div', {'class': 'hamburger icon button', events: {click: ev => { document.body.classList.toggle('droppeddown'); ev.stopPropagation(); ev.preventDefault(); }}}, create_element('div'), create_element('div'), create_element('div')));

  var header = document.getElementById('header');
  header.appendChild(create_element('div', {'class': 'dropdown', events: {click: ev => { ev.stopPropagation(); ev.preventDefault(); }}}, grid));

  window.addEventListener('keydown', keydown_handler);
  window.addEventListener('popstate', ev => apply_state(ev.state || {}));
  window.addEventListener('resize', ev => window.requestAnimationFrame(theater._update_overview_layout.bind(theater)));

  window.theater = new Theater();
  window.log = new Log();
  document.body.appendChild(theater.root);
  document.body.appendChild(create_element('div', {'class': 'dropdown-catchall', events: {click: ev => { document.body.classList.remove('droppeddown'); ev.stopPropagation(); ev.preventDefault(); }}}));

  const state = window.history.state || {};
  window.log.init_elements((state.log || {}).collapsed || {});
  if (state.log && Number.isInteger(state.log.loglevel))
    log.loglevel = state.log.loglevel;
  else
    log.loglevel = LEVELS.indexOf('info');
  apply_state(state);
  state_control = 'enabled';
});
