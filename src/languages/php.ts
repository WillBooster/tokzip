import { phpModule } from '../generated/php.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(phpModule);

export { phpModule };
