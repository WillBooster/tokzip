import { cssModule } from '../generated/css.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(cssModule);

export { cssModule };
