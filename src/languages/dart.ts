import { dartModule } from '../generated/dart.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(dartModule);

export { dartModule };
